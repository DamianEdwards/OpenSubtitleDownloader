using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.FileSystemGlobbing;
using Whisper.net;
using Whisper.net.Ggml;
using Whisper.net.Logger;

// TODO: Use System.CommandLine for command-line parsing
// TODO: Detect show name from directory name
// TODO: Find all seasons in the input folder
// TODO: Add some parallelism to speed up processing

var showName = "The Big Bang Theory";
//int season = 1;
int? episode = null;
var inputFolder = @$"G:\Video\{showName}";
var cacheDir = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".mkvmatchr");
var subtitlesFolder = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), $".subtitlr", showName);
var ffmpegPath = FindFfmpegPath() ?? throw new InvalidOperationException("ffmpeg not found on the PATH. Ensure ffmpeg is on the PATH and run again.");

var whatIf = true;
var ggmlType = GgmlType.LargeV3Turbo;
var chunkLength = TimeSpan.FromMinutes(5);
var sampleChunks = 3;

if (!Directory.Exists(inputFolder))
{
    WriteLine("Input folder not found.", ConsoleColor.Red);
    Environment.ExitCode = 1;
    return;
}

if (!Directory.Exists(subtitlesFolder))
{
    WriteLine("Subtitles folder not found.", ConsoleColor.Red);
    Environment.ExitCode = 2;
    return;
}

using var whisperFactory = await InitializeWhisper(cacheDir, ggmlType);

var subtitles = LoadSubtitles(showName, subtitlesFolder, chunkLength, sampleChunks);

// Print out what runtime Whisper is using
//WriteLine($"Whisper runtime: {WhisperFactory.GetRuntimeInfo()}", ConsoleColor.Gray);

var mkvFiles = GetMkvFiles(inputFolder);
foreach (var season in mkvFiles.Keys)
{
    var files = mkvFiles[season];
    var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = 1 };
    await Parallel.ForEachAsync(files, parallelOptions, async (filePath, ct) => await ProcessFile(filePath, season));
    //foreach (var filePath in files)
    //{
    //    await ProcessFile(filePath, season);
    //}
}

Dictionary<int, List<string>> GetMkvFiles(string inputFolder)
{
    var matcher = new Matcher();
    matcher.AddIncludePatterns(["Season*/*.mkv"]);

    var matchingFiles = matcher.GetResultsInFullPath(inputFolder);
    return matchingFiles
        .Select(p => (Season: ExtractSeasonFromFilePath(p) ?? -1, FilePath: p))
        .Where(p => p.Season > 0)
        .GroupBy(item => item.Season)
        .ToDictionary(g => g.Key, g => g.Select(item => item.FilePath).ToList());
}

static int? ExtractSeasonFromFilePath(string filePath)
{
    var match = DirectorySeasonRegex().Match(filePath);
    return match.Success && int.TryParse(match.Groups[1].Value, out var season) ? season : null;
}

async Task ProcessFile(string filePath, int season)
{
    var fileName = Path.GetFileName(filePath);

    if (episode is not null && !fileName.Contains($"S{season:D2}E{episode:D2}", StringComparison.OrdinalIgnoreCase))
    {
        WriteLine($"{fileName}: Skipping episode.", ConsoleColor.Gray);
        return;
    }

    WriteLine($"{fileName}: Processing file {filePath}");

    for (var i = 0; i < sampleChunks; i++)
    {
        var chunk = i + 1;
        var sampleLength = chunkLength * chunk;

        // Step 1: Extract audio from the .mkv file
        var audioPath = Path.ChangeExtension(filePath, ".wav");

        try
        {
            ExtractAudio(ffmpegPath, filePath, audioPath, sampleLength);

            // Step 2: Transcribe audio to text using Whisper
            var transcription = await TranscribeAudio(audioPath, season);

            // Step 3: Match transcription with subtitle files
            var bestMatch = FindBestMatchingSubtitle(filePath, showName, season, transcription, subtitles[season], chunk);

            if (bestMatch is not null)
            {
                // Step 4: Rename .mkv file
                RenameFile(showName, inputFolder, filePath, bestMatch);
                break;
            }

            if (i == sampleChunks - 1)
            {
                WriteLine($"{fileName}: No matching subtitle found.", ConsoleColor.Red);
            }
            else
            {
                WriteLine($"{fileName}: No matching subtitle found, increasing sample size to {chunkLength * (chunk + 1)}.", ConsoleColor.Blue);
            }
        }
        finally
        {
            // Clean up
            if (File.Exists(audioPath))
            {
                File.Delete(audioPath);
            }
        }
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
        var ffmpegPath = Path.Join(path, OperatingSystem.IsWindows() ? "ffmpeg.exe" : "ffmpeg");
        if (File.Exists(ffmpegPath))
        {
            return ffmpegPath;
        }
    }

    return null;
}

static void ExtractAudio(string ffmpegPath, string inputFile, string outputAudio, TimeSpan duration)
{
    var fileName = Path.GetFileName(inputFile);
    WriteLine($"{fileName}: Extracting audio from {fileName} to {Path.GetFileName(outputAudio)} ...");

    var startTime = "00:00:00";
    var time = duration.ToString(@"hh\:mm\:ss");
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

    WriteLine($"{fileName}: Audio extraction complete.");
}

// TODO: Pass in the StringBuilder and pool them at the call site
async Task<string> TranscribeAudio(string audioPath, int season)
{
    var fileName = Path.GetFileName(audioPath);
    WriteLine($"{fileName}: Transcribing audio from {fileName} ...");

    var sb = new StringBuilder();
    await using var processor = whisperFactory.CreateBuilder()
        .WithLanguage("en")
        .WithPrompt($"This is an episode of a TV show called {showName} from season {season}")
        .WithThreads(4)
        .Build();

    using var fileStream = File.OpenRead(audioPath);
    await foreach (var segment in processor.ProcessAsync(fileStream))
    {
        sb.Append(segment.Text);
    }

    WriteLine($"{fileName}: Transcribing done!");

    return sb.ToString();
}

static async Task DownloadModel(string filePath, GgmlType ggmlType)
{
    WriteLine($"Downloading model {Enum.GetName(ggmlType)} to {Path.GetFileName(filePath)}", ConsoleColor.Gray);

    using var modelStream = await WhisperGgmlDownloader.GetGgmlModelAsync(ggmlType);
    using var fileStream = File.OpenWrite(filePath);
    await modelStream.CopyToAsync(fileStream);
}

static string? FindBestMatchingSubtitle(string filePath, string showName, int season, string transcription, List<(string, string[])> subtitles, int chunk)
{
    WriteLine($"{Path.GetFileName(filePath)}: Finding best matching subtitle");

    string? bestMatch = null;
    double highestSimilarity = 0;

    foreach (var subtitle in subtitles)
    {
        var fileName = subtitle.Item1;
        var subtitleContent = string.Join(Environment.NewLine, subtitle.Item2.Take(chunk));

        var similarity = CalculateCosineSimilarity(transcription, subtitleContent);
        if (similarity > highestSimilarity)
        {
            highestSimilarity = similarity;
            bestMatch = fileName;
        }
    }

    if (highestSimilarity < 0.8)
    {
        WriteLine($"{Path.GetFileName(filePath)}: No matching subtitle above 80% found (highest was {highestSimilarity:P2}).", ConsoleColor.Yellow);
        return null;
    }

    WriteLine($"{Path.GetFileName(filePath)}: Match found! Best match is {Path.GetFileName(bestMatch)} with similarity of {highestSimilarity:P2}.", ConsoleColor.Green);
    return bestMatch;
}

static Dictionary<int, List<(string, string[])>> LoadSubtitles(string showName, string subtitlesFolder, TimeSpan duration, int chunkCount)
{
    // TODO: Change to use globbing via Microsoft.Extensions.FileSystemGlobbing
    var srtFiles = Directory.GetFiles(subtitlesFolder, "*.srt")
        .Select(file =>
        {
            var seasonParsed = int.TryParse(ExtractSeasonEpisode(file), out var season);
            return (Season: seasonParsed ? season : -1, FilePath: file);
        })
        .Where(file => file.Season >= 0)
        .ToArray();

    var result = new Dictionary<int, List<(string, string[])>>();

    foreach (var (season, filePath) in srtFiles)
    {
        if (!result.TryGetValue(season, out var seasonFiles))
        {
            seasonFiles = [];
            result[season] = seasonFiles;
        }

        var chunks = new string[chunkCount];
        for (var i = 0; i < chunkCount; i++)
        {
            var start = i * duration;
            chunks[i] = GetSubtitlesText(filePath, start, duration);
        }
        seasonFiles.Add((filePath, chunks));
    }

    WriteLine($"Loaded {srtFiles.Length} subtitle files for {result.Keys.Count} seasons from {subtitlesFolder}.", ConsoleColor.Gray);

    return result;
}

static string GetSubtitlesText(string filePath, TimeSpan startTime, TimeSpan duration)
{
    // Subtitle file will be in SRT format, e.g.:
    // 1
    // 00:00:00,552 --> 00:00:02,513
    // Here we go.
    // Pad thai, no peanuts.

    // 2
    // 00:00:02,633 --> 00:00:03,969
    // But does it have peanut oil?

    var sb = new StringBuilder();
    var endTime = startTime + duration;

    // Read all lines at once for better performance
    var lines = File.ReadAllLines(filePath);

    for (int i = 0; i < lines.Length; i++)
    {
        var line = lines[i];

        // Skip empty lines
        if (string.IsNullOrWhiteSpace(line))
        {
            continue;
        }

        // Skip index number
        if (int.TryParse(line, out var _))
        {
            continue;
        }

        // Parse timestamp line

        // Extract start and end times
        var timestamps = line.Split(" --> ");
        if (timestamps.Length != 2)
        {
            continue;
        }

        var start = timestamps[0].Replace(',', '.');

        if (TimeSpan.TryParse(start, out var startTimeStamp))
        {
            // Skip if we're before our start time
            if (startTimeStamp < startTime)
            {
                continue;
            }

            // Skip if we're past our end time
            if (startTimeStamp > endTime)
            {
                break;
            }

            // Advance to next line
            i++;
        }

        // Collect subtitle text until empty line
        while (i < lines.Length)
        {
            line = lines[i];
            if (string.IsNullOrWhiteSpace(line))
            {
                break;
            }

            sb.AppendLine(HtmlTags().Replace(line, string.Empty));
            i++;
        }
    }

    return sb.ToString().Trim();
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

    static void GetWordCounts(string text1, Dictionary<string, int> wordCounts, HashSet<string> uniqueWords)
    {
        foreach (var word in text1.AsSpan().EnumerateLines())
        {
            // TODO: Avoid using ToString() and Split() here to improve performance
            foreach (var splitWord in word.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                var key = splitWord.ToLowerInvariant();
                // TODO: Use span-based alternate dictionary lookup
                if (wordCounts.TryGetValue(key, out var count))
                {
                    wordCounts[key] = count + 1;
                }
                else
                {
                    wordCounts[key] = 1;
                    uniqueWords.Add(key);
                }
            }
        }
    }
}

static string? ExtractSeasonEpisode(string fileName)
{
    var match = SeasonEpisodeRegex().Match(Path.GetFileNameWithoutExtension(fileName));
    return match.Success ? match.Groups[1].Value : null;
}

void RenameFile(string showName, string inputFolder, string currentFilePath, string bestMatch)
{
    var seasonEpisode = ExtractSeasonEpisode(bestMatch);
    if (!string.IsNullOrEmpty(seasonEpisode))
    {
        var newFileName = $"{showName} - {seasonEpisode}.mkv";
        var newFilePath = Path.Join(inputFolder, newFileName);

        if (File.Exists(newFilePath))
        {
            // TODO: Handle this case better, maybe rename existing file to a different name
            WriteLine($"File already exists: {newFileName}", ConsoleColor.Blue);
            return;
        }

        if (whatIf)
        {
            WriteLine($"Would rename to: {newFileName}", ConsoleColor.Yellow);
        }
        else
        {
            File.Move(currentFilePath, newFilePath);
            WriteLine($"Renamed to: {newFileName}", ConsoleColor.Green);
        }
    }
}

static void WriteLine(string line, ConsoleColor? color = null)
{
    if (color is null)
    {
        Console.WriteLine(line);
        return;
    }

    lock (ConsoleWritingLock)
    {
        var existingColor = Console.ForegroundColor;
        Console.ForegroundColor = color.Value;
        Console.WriteLine(line);
        Console.ForegroundColor = existingColor;
    }
}

static async Task<WhisperFactory> InitializeWhisper(string cacheDir, GgmlType ggmlType)
{
    using var whisperLogger = LogProvider.AddConsoleLogging(WhisperLogLevel.Info);

    var modelFileName = $"ggml-{Enum.GetName(ggmlType)!.ToLowerInvariant()}.bin";
    var modelFilePath = Path.Join(cacheDir, "whisper", modelFileName);

    if (!File.Exists(modelFilePath))
    {
        await DownloadModel(modelFilePath, ggmlType);
    }

    var whisperFactory = WhisperFactory.FromPath(modelFilePath);
    return whisperFactory;
}

partial class Program
{
    public static Lock ConsoleWritingLock = new();

    [GeneratedRegex(@"s(\d{2})e(\d{2})", RegexOptions.IgnoreCase, "en-US")]
    public static partial Regex SeasonEpisodeRegex();

    [GeneratedRegex(@"Season\s+(\d+)", RegexOptions.IgnoreCase, "en-US")]
    private static partial Regex DirectorySeasonRegex();

    [GeneratedRegex(@"\d+\r?\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\r?\n")]
    public static partial Regex SrtTimeStamp();

    [GeneratedRegex(@"<[^>]+>")]
    public static partial Regex HtmlTags();
}
