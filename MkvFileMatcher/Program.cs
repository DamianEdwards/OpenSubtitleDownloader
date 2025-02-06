using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using Whisper.net;
using Whisper.net.Ggml;
using Whisper.net.Logger;

// TODO: Use System.CommandLine for command-line parsing
// TODO: Detect show name from directory name
// TODO: Find all seasons in the input folder
// TODO: Add some parallelism to speed up processing

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

// TODO: Change to use globbing via Microsoft.Extensions.FileSystemGlobbing
foreach (var filePath in Directory.GetFiles(inputFolder, "*.mkv"))
{
    Console.WriteLine($"Processing: {Path.GetFileName(filePath)}");

    // Step 1: Extract audio from the .mkv file
    var audioPath = Path.ChangeExtension(filePath, ".wav");
    ExtractAudio(ffmpegPath, filePath, audioPath);

    // Step 2: Transcribe audio to text using Whisper
    var transcription = await TranscribeAudio(audioPath);

    // Step 3: Match transcription with subtitle files
    var bestMatch = FindBestMatchingSubtitle(showName, season, transcription, subtitlesFolder);

    if (bestMatch is not null)
    {
        // Step 4: Rename .mkv file
        RenameFile(showName, inputFolder, bestMatch);
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
    Console.Write($"Extracting audio from {Path.GetFileName(inputFile)} to {Path.GetFileName(outputAudio)} ...");

    var startTime = "00:00:00";
    var audioLengthMins = TimeSpan.FromMinutes(5);
    var time = audioLengthMins.ToString(@"hh\:mm\:ss");
    var codec = "pcm_s16le";
    var samplingRate = 16000;
    var channels = 1; // mono
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

    Console.WriteLine(" done!");
}

async Task<string> TranscribeAudio(string audioPath)
{
    Console.Write($"Transcribing audio from {Path.GetFileName(audioPath)} ...");

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

    using var fileStream = File.OpenRead(audioPath);
    processor.Process(fileStream);

    Console.WriteLine(" done!");

    return sb.ToString();
}

static async Task DownloadModel(string fileName, GgmlType ggmlType)
{
    Console.WriteLine($"Downloading Model {fileName}");
    using var modelStream = await WhisperGgmlDownloader.GetGgmlModelAsync(ggmlType);
    using var fileStream = File.OpenWrite(fileName);
    await modelStream.CopyToAsync(fileStream);
}

// TODO: Cache the cleaned subtitle content so this isn't done on every call, e.g. just do it up front
static string? FindBestMatchingSubtitle(string showName, int season, string transcription, string subtitlesFolder)
{
    Console.Write("Finding best matching subtitle ...");

    var fileNamePrefix = $"{showName} - S{season:D2}";
    // TODO: Change to use globbing via Microsoft.Extensions.FileSystemGlobbing
    var srtFiles = Directory.GetFiles(subtitlesFolder, "*.srt")
        .Where(file => Path.GetFileName(file).StartsWith(fileNamePrefix, StringComparison.OrdinalIgnoreCase))
        .ToArray();

    string? bestMatch = null;
    double highestSimilarity = 0;

    foreach (var file in srtFiles)
    {
        var subtitleContent = File.ReadAllText(file);
        // Remove SRT timestamps and numbers using regex
        var cleanedContent = SrtTimeStamp().Replace(subtitleContent, string.Empty);
        cleanedContent = HtmlTags().Replace(cleanedContent, string.Empty); // Remove HTML tags

        var similarity = CalculateCosineSimilarity(transcription, cleanedContent);
        if (similarity > highestSimilarity)
        {
            highestSimilarity = similarity;
            bestMatch = file;
        }
    }

    Console.WriteLine($" done! Best match is {Path.GetFileName(bestMatch)} with similarity of {highestSimilarity:P2}.");

    return bestMatch;
}

static double CalculateCosineSimilarity(string text1, string text2)
{
    // Create word frequency dictionaries with initial capacity
    // TODO: Reuse these dictionaries across calls
    var words1 = new Dictionary<string, int>(1024);
    var words2 = new Dictionary<string, int>(1024);
    var uniqueWords = new HashSet<string>(1024);

    // Process first text
    GetWordCounts(text1, words1, uniqueWords);

    // Process second text
    GetWordCounts(text2, words2, uniqueWords);

    // Calculate similarity in a single pass
    double dotProduct = 0, magnitude1 = 0, magnitude2 = 0;

    foreach (var word in uniqueWords)
    {
        var value1 = words1.GetValueOrDefault(word);
        var value2 = words2.GetValueOrDefault(word);
        dotProduct += value1 * value2;
        magnitude1 += value1 * value1;
        magnitude2 += value2 * value2;
    }

    magnitude1 = Math.Sqrt(magnitude1);
    magnitude2 = Math.Sqrt(magnitude2);

    return (magnitude1 > 0 && magnitude2 > 0) ? dotProduct / (magnitude1 * magnitude2) : 0;

    static void GetWordCounts(string text1, Dictionary<string, int> words1, HashSet<string> uniqueWords)
    {
        foreach (var word in text1.AsSpan().EnumerateLines())
        {
            foreach (var splitWord in word.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                var key = splitWord.ToLowerInvariant();
                if (words1.TryGetValue(key, out var count))
                {
                    words1[key] = count + 1;
                }
                else
                {
                    words1[key] = 1;
                    uniqueWords.Add(key);
                }
            }
        }
    }
}

string? ExtractSeasonEpisode(string subtitleFileName)
{
    var match = SeasonEpisodeRegex().Match(Path.GetFileNameWithoutExtension(subtitleFileName));
    return match.Success ? match.Value : null;
}

void RenameFile(string showName, string inputFolder, string bestMatch)
{
    var seasonEpisode = ExtractSeasonEpisode(bestMatch);
    if (!string.IsNullOrEmpty(seasonEpisode))
    {
        var newFileName = $"{showName} - {seasonEpisode}.mkv";
        var newFilePath = Path.Combine(inputFolder, newFileName);

        if (File.Exists(newFilePath))
        {
            // TODO: Handle this case better, maybe rename existing file to a different name
            Console.WriteLine($"File already exists: {newFileName}");
            return;
        }

        Console.WriteLine($"Would rename to: {newFileName}");
        //File.Move(filePath, newFilePath);
        //Console.WriteLine($"Renamed to: {newFileName}");
    }
}

partial class Program
{
    [GeneratedRegex(@"s(\d{2})e(\d{2})", RegexOptions.IgnoreCase, "en-US")]
    public static partial Regex SeasonEpisodeRegex();

    [GeneratedRegex(@"\d+\r?\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\r?\n")]
    public static partial Regex SrtTimeStamp();

    [GeneratedRegex(@"<[^>]+>")]
    public static partial Regex HtmlTags();
}
