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

var showName = "Young Sheldon";
int season = 2;
int? episode = null;
var inputFolder = @$"G:\Video\{showName}\Season {season}";
var cacheDir = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".mkvmatchr");
var subtitlesFolder = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), $".subtitlr", showName);
var ffmpegPath = FindFfmpegPath() ?? throw new InvalidOperationException("ffmpeg not found on the PATH. Ensure ffmpeg is on the PATH and run again.");

var whatIf = false;
var ggmlType = GgmlType.MediumEn;
var chunkLength = TimeSpan.FromMinutes(5);
var sampleChunks = 3;

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

var subtitles = LoadSubtitles(showName, season, subtitlesFolder, chunkLength, sampleChunks);

// Print out what runtime Whisper is using
//Console.WriteLine($"Whisper runtime: {WhisperFactory.GetRuntimeInfo()}");

// TODO: Change to use globbing via Microsoft.Extensions.FileSystemGlobbing
foreach (var filePath in Directory.GetFiles(inputFolder, "*.mkv"))
{
    Console.WriteLine($"Processing: {Path.GetFileName(filePath)}");

    if (episode is not null && !Path.GetFileName(filePath).Contains($"S{season:D2}E{episode:D2}", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine("Skipping episode.");
        continue;
    }

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
            var transcription = await TranscribeAudio(audioPath);

            // Step 3: Match transcription with subtitle files
            var bestMatch = FindBestMatchingSubtitle(showName, season, transcription, subtitles, chunk);

            if (bestMatch is not null)
            {
                // Step 4: Rename .mkv file
                RenameFile(showName, inputFolder, filePath, bestMatch);
                break;
            }

            if (i == sampleChunks - 1)
            {
                Console.WriteLine("No matching subtitle found.");
            }
            else
            {
                Console.WriteLine($"No matching subtitle found, increasing sample size to {chunkLength * (chunk + 1)}.");
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

void ExtractAudio(string ffmpegPath, string inputFile, string outputAudio, TimeSpan duration)
{
    Console.Write($"Extracting audio from {Path.GetFileName(inputFile)} to {Path.GetFileName(outputAudio)} ...");

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

    Console.WriteLine(" done!");
}

async Task<string> TranscribeAudio(string audioPath)
{
    Console.Write($"Transcribing audio from {Path.GetFileName(audioPath)} ...");

    using var whisperLogger = LogProvider.AddConsoleLogging(WhisperLogLevel.Info);
    
    var modelFileName = $"ggml-{Enum.GetName(ggmlType)!.ToLowerInvariant()}.bin";
    var modelFilePath = Path.Combine(cacheDir, "whisper", modelFileName);

    if (!File.Exists(modelFilePath))
    {
        await DownloadModel(modelFilePath, ggmlType);
    }

    using var whisperFactory = WhisperFactory.FromPath(modelFilePath);

    var sb = new StringBuilder();
    using var processor = whisperFactory.CreateBuilder()
        .WithLanguage("en")
        .WithPrompt($"This is an episode of a TV show called {showName} from season {season}")
        .WithSegmentEventHandler((segment) => sb.Append(segment.Text))
        .Build();

    using var fileStream = File.OpenRead(audioPath);
    processor.Process(fileStream);

    Console.WriteLine(" done!");

    return sb.ToString();
}

static async Task DownloadModel(string filePath, GgmlType ggmlType)
{
    Console.WriteLine($"Downloading model {Enum.GetName(ggmlType)} to {Path.GetFileName(filePath)}");

    using var modelStream = await WhisperGgmlDownloader.GetGgmlModelAsync(ggmlType);
    using var fileStream = File.OpenWrite(filePath);
    await modelStream.CopyToAsync(fileStream);
}

static string? FindBestMatchingSubtitle(string showName, int season, string transcription, List<(string, string[])> subtitles, int chunk)
{
    Console.Write("Finding best matching subtitle ...");

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
        Console.WriteLine(" no match above 80% found.");
        return null;
    }

    Console.WriteLine($" done! Best match is {Path.GetFileName(bestMatch)} with similarity of {highestSimilarity:P2}.");
    return bestMatch;
}

static List<(string, string[])> LoadSubtitles(string showName, int season, string subtitlesFolder, TimeSpan duration, int chunkCount)
{
    var fileNamePrefix = $"{showName} - S{season:D2}";

    // TODO: Change to use globbing via Microsoft.Extensions.FileSystemGlobbing
    var srtFiles = Directory.GetFiles(subtitlesFolder, "*.srt")
        .Where(file => Path.GetFileName(file).StartsWith(fileNamePrefix, StringComparison.OrdinalIgnoreCase))
        .ToArray();

    var result = new List<(string, string[])>(srtFiles.Length);

    foreach (var file in srtFiles)
    {
        var chunks = new string[chunkCount];
        for (var i = 0; i < chunkCount; i++)
        {
            var start = i * duration;
            chunks[i] = GetSubtitlesText(file, start, duration);
        }
        result.Add((file, chunks));
    }

    Console.WriteLine($"Loaded {result.Count} subtitle files from {subtitlesFolder}.");

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

string? ExtractSeasonEpisode(string subtitleFileName)
{
    var match = SeasonEpisodeRegex().Match(Path.GetFileNameWithoutExtension(subtitleFileName));
    return match.Success ? match.Value : null;
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
            Console.WriteLine($"File already exists: {newFileName}");
            return;
        }

        if (whatIf)
        {
            Console.WriteLine($"Would rename to: {newFileName}");
        }
        else
        {
            File.Move(currentFilePath, newFilePath);
            Console.WriteLine($"Renamed to: {newFileName}");
        }
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
