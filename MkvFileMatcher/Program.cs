using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Whisper.net;
using Whisper.net.Ggml;

// Paths
string inputFolder = @"path\to\input\mkv\files";
string subtitlesFolder = @"path\to\subtitle\files";
string ffmpegPath = @"path\to\ffmpeg.exe";

string showName = "The Big Bang Theory";

var ggmlType = GgmlType.Base;
var modelFileName = "ggml-base.bin";

foreach (var filePath in Directory.GetFiles(inputFolder, "*.mkv"))
{
    Console.WriteLine($"Processing: {Path.GetFileName(filePath)}");

    // Step 1: Extract audio from the .mkv file
    var audioPath = Path.ChangeExtension(filePath, ".wav");
    ExtractAudio(ffmpegPath, filePath, audioPath);

    // Step 2: Transcribe audio to text using Whisper
    var transcription = await TranscribeAudio(audioPath, durationLimit: 300);

    // Step 3: Match transcription with subtitle files using ML.NET
    var bestMatch = FindBestMatchingSubtitleWithML(transcription, subtitlesFolder);

    if (bestMatch != null)
    {
        // Step 4: Rename .mkv file
        var seasonEpisode = ExtractSeasonEpisode(bestMatch);
        if (!string.IsNullOrEmpty(seasonEpisode))
        {
            var newFileName = $"{showName} - {seasonEpisode}.mkv";
            var newFilePath = Path.Combine(inputFolder, newFileName);
            File.Move(filePath, newFilePath);
            Console.WriteLine($"Renamed to: {newFileName}");
        }
    }
    else
    {
        Console.WriteLine("No matching subtitle found.");
    }

    // Clean up
    if (File.Exists(audioPath)) File.Delete(audioPath);
}

void ExtractAudio(string ffmpegPath, string inputFile, string outputAudio)
{
    var process = new Process
    {
        StartInfo = new ProcessStartInfo
        {
            FileName = ffmpegPath,
            Arguments = $"-i \"{inputFile}\" -q:a 0 -map a \"{outputAudio}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        }
    };

    process.Start();
    process.WaitForExit();

    if (process.ExitCode != 0)
    {
        throw new Exception($"FFmpeg failed: {process.StandardError.ReadToEnd()}");
    }
}

async Task<string> TranscribeAudio(string audioPath, int durationLimit = 0)
{
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

    var audioData = TrimAudio(audioPath, durationLimit);
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

static string? FindBestMatchingSubtitleWithML(string transcription, string subtitlesFolder)
{
    // Load subtitles into ML.NET data view
    var mlContext = new MLContext();
    var subtitleFiles = Directory.GetFiles(subtitlesFolder, "*.srt").Select(file => new SubtitleData
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

byte[] TrimAudio(string audioPath, int durationLimit)
{
    string tempFilePath = Path.ChangeExtension(Path.GetTempFileName(), ".wav");

    var process = new Process
    {
        StartInfo = new ProcessStartInfo
        {
            FileName = ffmpegPath,
            Arguments = $"-i \"{audioPath}\" -t {durationLimit} -c copy \"{tempFilePath}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        }
    };

    process.Start();
    process.WaitForExit();

    if (process.ExitCode != 0)
    {
        throw new Exception($"FFmpeg failed: {process.StandardError.ReadToEnd()}");
    }

    byte[] trimmedAudio = File.ReadAllBytes(tempFilePath);
    File.Delete(tempFilePath);

    return trimmedAudio;
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