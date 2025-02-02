using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;
using Whisper.net;
using Whisper.net.Ggml;
using Whisper.net.Logger;

// Paths
string showName = "Young Sheldon";
int season = 1;
string inputFolder = @$"G:\Video\{showName}\Season {season}";
string subtitlesFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), $".subtitlr", showName);
string ffmpegPath = FindFfmpegPath() ?? throw new InvalidOperationException("ffmpeg not found on the PATH. Ensure ffmpeg is on the PATH and run again.");

var ggmlType = GgmlType.Base;
var modelFileName = "ggml-base.bin";

if (!Directory.Exists(inputFolder))
{
    Console.WriteLine("Input folder not found.");
    Environment.ExitCode = 1;
    return;
}

if (!Directory.Exists(subtitlesFolder))
{
    Console.WriteLine("Subtitles folder not found.");
    Environment.ExitCode = 2;
    return;
}

// Print out what runtime Whisper is using
Console.WriteLine($"Whisper runtime: {WhisperFactory.GetRuntimeInfo()}");

foreach (var filePath in Directory.GetFiles(inputFolder, "*.mkv"))
{
    Console.WriteLine($"Processing: {Path.GetFileName(filePath)}");

    // Step 1: Extract audio from the .mkv file
    var audioPath = Path.ChangeExtension(filePath, ".wav");
    ExtractAudio(ffmpegPath, filePath, audioPath);

    // Step 2: Transcribe audio to text using Whisper
    var transcription = await TranscribeAudio(audioPath);

    // Step 3: Match transcription with subtitle files using ML.NET
    var bestMatch = FindBestMatchingSubtitleWithML(showName, season, transcription, subtitlesFolder);

    if (bestMatch != null)
    {
        // Step 4: Rename .mkv file
        var seasonEpisode = ExtractSeasonEpisode(bestMatch);
        if (!string.IsNullOrEmpty(seasonEpisode))
        {
            var newFileName = $"{showName} - {seasonEpisode}.mkv";
            var newFilePath = Path.Combine(inputFolder, newFileName);
            if (File.Exists(newFilePath))
            {
                Console.WriteLine($"File already exists: {newFileName}");
                continue;
            }
            Console.WriteLine($"Would rename to: {newFileName}");
            //File.Move(filePath, newFilePath);
            //Console.WriteLine($"Renamed to: {newFileName}");
        }
    }
    else
    {
        Console.WriteLine("No matching subtitle found.");
    }

    // Clean up
    if (File.Exists(audioPath))
    {
        File.Delete(audioPath);
    }
}

string? FindFfmpegPath()
{
    var paths = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator);
    if (paths is null)
    {
        return null;
    }

    foreach (var path in paths)
    {
        var ffmpegPath = Path.Combine(path, OperatingSystem.IsWindows() ? "ffmpeg.exe" : "ffmpeg");
        if (File.Exists(ffmpegPath))
        {
            return ffmpegPath;
        }
    }

    return null;
}

void ExtractAudio(string ffmpegPath, string inputFile, string outputAudio)
{
    var startTime = "00:00:00";
    var audioLengthMins = TimeSpan.FromMinutes(5);
    var time = audioLengthMins.ToString(@"hh\:mm\:ss");
    var codec = "pcm_s16le";
    var samplingRate = 16000;
    var channels = 1;
    var process = new Process
    {
        StartInfo = new ProcessStartInfo
        {
            FileName = ffmpegPath,
            Arguments = $"-i \"{inputFile}\" -y -ss {startTime} -t {time} -vn -acodec {codec} -ar {samplingRate} -ac {channels} \"{outputAudio}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        }
    };

    var outputBuilder = new StringBuilder();
    var errorBuilder = new StringBuilder();

    process.OutputDataReceived += (sender, args) => outputBuilder.AppendLine(args.Data);
    process.ErrorDataReceived += (sender, args) => errorBuilder.AppendLine(args.Data);

    process.Start();
    process.BeginOutputReadLine();
    process.BeginErrorReadLine();

    if (!process.WaitForExit(TimeSpan.FromSeconds(30)))
    {
        process.Kill();
        throw new Exception($"FFmpeg process timed out: {outputBuilder}");
    }

    if (process.ExitCode != 0)
    {
        throw new Exception($"FFmpeg failed: {errorBuilder}");
    }
}

async Task<string> TranscribeAudio(string audioPath)
{
    using var whisperLogger = LogProvider.AddConsoleLogging(WhisperLogLevel.Info);

    if (!File.Exists(modelFileName))
    {
        await DownloadModel(modelFileName, ggmlType);
    }

    using var whisperFactory = WhisperFactory.FromPath(modelFileName);

    var sb = new StringBuilder();
    using var processor = whisperFactory.CreateBuilder()
        .WithLanguage("en")
        .WithSegmentEventHandler((segment) => sb.Append(segment.Text))
        .Build();

    byte[] audioData = File.ReadAllBytes(audioPath);
    using var ms = new MemoryStream(audioData);

    processor.Process(ms);

    return sb.ToString();
}

static async Task DownloadModel(string fileName, GgmlType ggmlType)
{
    Console.WriteLine($"Downloading Model {fileName}");
    using var modelStream = await WhisperGgmlDownloader.GetGgmlModelAsync(ggmlType);
    using var fileWriter = File.OpenWrite(fileName);
    await modelStream.CopyToAsync(fileWriter);
}

static string? FindBestMatchingSubtitleWithML(string showName, int season, string transcription, string subtitlesFolder)
{
    // Load subtitles into ML.NET data view
    var mlContext = new MLContext();
    var fileNamePrefix = $"{showName} - S{season:D2}";
    var srtFiles = Directory.GetFiles(subtitlesFolder, "*.srt")
        .Where(file => Path.GetFileName(file).StartsWith(fileNamePrefix, StringComparison.OrdinalIgnoreCase))
        .ToArray();
    var subtitleFiles = srtFiles.Select(file => new SubtitleData
    {
        FilePath = file,
        Content = File.ReadAllText(file)
    }).ToList();

    // TODO: Remove the timestamps from the subtitle content

    var dataView = mlContext.Data.LoadFromEnumerable(subtitleFiles);

    // Define pipeline
    var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SubtitleData.Content))
        .Append(mlContext.Transforms.Concatenate("Features", "Features"))
        .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

    // Train model if needed
    var modelFilePath = "model.zip";
    ITransformer model;

    if (File.Exists(modelFilePath))
    {
        using var stream = File.OpenRead(modelFilePath);
        model = mlContext.Model.Load(stream, out _);
    }
    else
    {
        model = pipeline.Fit(dataView);
        using var stream = File.Create(modelFilePath);
        mlContext.Model.Save(model, dataView.Schema, stream);
    }

    // Create prediction engine
    var predictionEngine = mlContext.Model.CreatePredictionEngine<SubtitleData, SubtitlePrediction>(model);

    // Find best match
    string? bestMatch = null;
    double highestScore = double.MinValue;

    foreach (var subtitle in subtitleFiles)
    {
        var prediction = predictionEngine.Predict(new SubtitleData { Content = transcription });
        if (prediction.Score > highestScore)
        {
            highestScore = prediction.Score;
            bestMatch = subtitle.FilePath;
        }
    }

    return bestMatch;
}

string? ExtractSeasonEpisode(string subtitleFileName)
{
    var match = SeasonEpisodeRegex().Match(Path.GetFileNameWithoutExtension(subtitleFileName));
    return match.Success ? match.Value : null;
}

public class SubtitleData
{
    public string? FilePath { get; set; }
    public required string Content { get; set; }
}

public class SubtitlePrediction
{
    [ColumnName("Score")]
    public float Score { get; set; }
}

partial class Program
{
    [GeneratedRegex(@"s(\d{2})e(\d{2})", RegexOptions.IgnoreCase, "en-US")]
    public static partial Regex SeasonEpisodeRegex();
}