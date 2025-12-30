using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.FileSystemGlobbing;
using Nikse.SubtitleEdit.Core.BluRaySup;
using Tesseract;
using Whisper.net;
using Whisper.net.Ggml;
using Whisper.net.Logger;

// TODO: Use System.CommandLine for command-line parsing
// TODO: Detect show name from directory name
// TODO: Find all seasons in the input folder
// TODO: Add some parallelism to speed up processing

var showName = "Brooklyn Nine-Nine";
int? specificSeason = 8; // Leave as null to process all seasons
int? episode = null; // Used for debugging specific episode
var inputFolder = @$"G:\Video\{showName}";
var cacheDir = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".mkvmatchr");
var subtitlesFolder = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), $".subtitlr", showName);
var ffmpegPath = FindFfmpegPath() ?? throw new InvalidOperationException("ffmpeg not found on the PATH. Ensure ffmpeg is on the PATH and run again.");
var tessdataPath = Path.Join(cacheDir, "tessdata");

var whatIf = args.Contains("--dry-run", StringComparer.OrdinalIgnoreCase);
var ggmlType = GgmlType.LargeV2;
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

// Ensure Tesseract data is available for PGS subtitle OCR
await EnsureTessdataExists(tessdataPath);

// Lazy-load Whisper only if audio transcription fallback is needed
WhisperFactory? whisperFactory = null;
IDisposable? whisperLogger = null;
async Task<WhisperFactory> GetWhisperFactory()
{
    if (whisperFactory is null)
    {
        whisperLogger = LogProvider.AddConsoleLogging(WhisperLogLevel.Info);
        whisperFactory = await InitializeWhisper(cacheDir, ggmlType);
    }
    return whisperFactory;
}

var subtitles = LoadSubtitles(showName, subtitlesFolder, chunkLength, sampleChunks);

// Print out what runtime Whisper is using
//WriteLine($"Whisper runtime: {WhisperFactory.GetRuntimeInfo()}", ConsoleColor.Gray);

var mkvFiles = GetMkvFiles(inputFolder);
foreach (var season in mkvFiles.Keys)
{
    if (specificSeason is not null && season != specificSeason.Value)
    {
        continue;
    }

    var files = mkvFiles[season];
    var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = 1 };
    await Parallel.ForEachAsync(files, parallelOptions, async (filePath, ct) => await ProcessFile(filePath, season));
    //foreach (var filePath in files)
    //{
    //    await ProcessFile(filePath, season);
    //}
}

// Dispose Whisper factory and logger if they were initialized
whisperFactory?.Dispose();
whisperLogger?.Dispose();

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

    // First, try to extract embedded subtitles from the MKV file
    var embeddedTextSubtitlePath = Path.ChangeExtension(filePath, ".embedded.srt");
    var embeddedSupPath = Path.ChangeExtension(filePath, ".embedded.sup");
    var (hasTextSubs, hasImageSubs, imageFormat) = TryExtractEmbeddedSubtitles(ffmpegPath, filePath, embeddedTextSubtitlePath, embeddedSupPath);

    try
    {
        // Priority 1: Use text-based embedded subtitles (SRT, ASS, etc.)
        if (hasTextSubs)
        {
            WriteLine($"{fileName}: Found text-based embedded subtitles, using them for matching.", ConsoleColor.Cyan);

            if (await TryMatchWithTextSubtitles(filePath, season, fileName, embeddedTextSubtitlePath))
            {
                return;
            }

            WriteLine($"{fileName}: Text-based embedded subtitles did not produce a match.", ConsoleColor.Yellow);
        }

        // Priority 2: Use image-based embedded subtitles with OCR (PGS, etc.)
        if (hasImageSubs)
        {
            WriteLine($"{fileName}: Found image-based subtitles ({imageFormat}), using OCR for matching.", ConsoleColor.Cyan);

            if (await TryMatchWithOcrSubtitles(filePath, season, fileName, embeddedSupPath))
            {
                return;
            }

            WriteLine($"{fileName}: OCR-based matching did not produce a match.", ConsoleColor.Yellow);
        }

        // Priority 3: Fallback to audio transcription
        if (!hasTextSubs && !hasImageSubs)
        {
            WriteLine($"{fileName}: No embedded subtitles found, using audio transcription.", ConsoleColor.Yellow);
        }
        else
        {
            WriteLine($"{fileName}: Falling back to audio transcription.", ConsoleColor.Yellow);
        }

        await ProcessFileWithAudioTranscription(filePath, season, fileName);
    }
    finally
    {
        // Clean up extracted subtitle files
        if (File.Exists(embeddedTextSubtitlePath))
        {
            File.Delete(embeddedTextSubtitlePath);
        }
        if (File.Exists(embeddedSupPath))
        {
            File.Delete(embeddedSupPath);
        }
    }
}

async Task<bool> TryMatchWithTextSubtitles(string filePath, int season, string fileName, string subtitlePath)
{
    for (var i = 0; i < sampleChunks; i++)
    {
        var chunk = i + 1;
        var sampleLength = chunkLength * chunk;

        // Get text from embedded subtitles
        var embeddedText = GetSubtitlesText(subtitlePath, TimeSpan.Zero, sampleLength);

        // Match embedded subtitles with downloaded subtitle files
        var bestMatch = FindBestMatchingSubtitle(filePath, showName, season, embeddedText, subtitles[season], chunk);

        if (bestMatch is not null)
        {
            RenameFile(showName, season, inputFolder, filePath, bestMatch);
            return true;
        }

        if (i < sampleChunks - 1)
        {
            WriteLine($"{fileName}: No matching subtitle found, increasing sample size to {chunkLength * (chunk + 1)}.", ConsoleColor.Blue);
        }
    }

    return false;
}

async Task<bool> TryMatchWithOcrSubtitles(string filePath, int season, string fileName, string supPath)
{
    try
    {
        // Parse the SUP file using libse
        var log = new StringBuilder();
        var pgsSubtitles = BluRaySupParser.ParseBluRaySup(supPath, log);

        if (pgsSubtitles.Count == 0)
        {
            WriteLine($"{fileName}: No PGS subtitles found in the extracted file.", ConsoleColor.Yellow);
            return false;
        }

        WriteLine($"{fileName}: Found {pgsSubtitles.Count} PGS subtitle frames, performing OCR...", ConsoleColor.Gray);

        // OCR the subtitle images
        var ocrText = await OcrPgsSubtitles(pgsSubtitles, fileName, chunkLength * sampleChunks);

        if (string.IsNullOrWhiteSpace(ocrText))
        {
            WriteLine($"{fileName}: OCR produced no text.", ConsoleColor.Yellow);
            return false;
        }

        // Try matching with different chunk sizes
        for (var i = 0; i < sampleChunks; i++)
        {
            var chunk = i + 1;
            var sampleLength = chunkLength * chunk;

            // Get OCR text up to the sample length
            var ocrTextForDuration = GetOcrTextForDuration(pgsSubtitles, ocrText, sampleLength);

            // Match OCR'd subtitles with downloaded subtitle files (using lower threshold for OCR)
            var bestMatch = FindBestMatchingSubtitle(filePath, showName, season, ocrTextForDuration, subtitles[season], chunk, isOcrText: true);

            if (bestMatch is not null)
            {
                RenameFile(showName, season, inputFolder, filePath, bestMatch);
                return true;
            }

            if (i < sampleChunks - 1)
            {
                WriteLine($"{fileName}: No matching subtitle found, increasing sample size to {chunkLength * (chunk + 1)}.", ConsoleColor.Blue);
            }
        }

        return false;
    }
    catch (Exception ex)
    {
        WriteLine($"{fileName}: Error during OCR processing: {ex.Message}", ConsoleColor.Red);
        return false;
    }
}

async Task<string> OcrPgsSubtitles(List<BluRaySupParser.PcsData> pgsSubtitles, string fileName, TimeSpan maxDuration)
{
    // Filter subtitles within duration and prepare for parallel processing
    var subtitlesToProcess = pgsSubtitles
        .Where(s => s.StartTimeCode.TotalMilliseconds <= maxDuration.TotalMilliseconds)
        .ToList();

    if (subtitlesToProcess.Count == 0)
    {
        return string.Empty;
    }

    // Step 1: Extract all bitmaps to PNG bytes in parallel (this is CPU-bound and thread-safe)
    var bitmapTasks = subtitlesToProcess
        .Select((subtitle, index) => Task.Run(() =>
        {
            try
            {
                using var bitmap = subtitle.GetBitmap();
                if (bitmap.Width <= 1 || bitmap.Height <= 1)
                {
                    return (Index: index, PngBytes: (byte[]?)null);
                }

                using var ms = new MemoryStream();
                bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                return (Index: index, PngBytes: (byte[]?)ms.ToArray());
            }
            catch
            {
                return (Index: index, PngBytes: (byte[]?)null);
            }
        }))
        .ToList();

    var bitmapResults = await Task.WhenAll(bitmapTasks);
    var validBitmaps = bitmapResults
        .Where(r => r.PngBytes is not null)
        .OrderBy(r => r.Index)
        .ToList();

    if (validBitmaps.Count == 0)
    {
        return string.Empty;
    }

    // Step 2: OCR the bitmaps in parallel using multiple Tesseract engines
    var degreeOfParallelism = Math.Min(Environment.ProcessorCount, validBitmaps.Count);
    var results = new string?[validBitmaps.Count];

    try
    {
        await Parallel.ForEachAsync(
            validBitmaps.Select((b, i) => (Bitmap: b, ResultIndex: i)),
            new ParallelOptions { MaxDegreeOfParallelism = degreeOfParallelism },
            async (item, ct) =>
            {
                // Each parallel task gets its own Tesseract engine (engines are not thread-safe)
                using var engine = new TesseractEngine(tessdataPath, "eng", EngineMode.Default);
                // Suppress "Empty page!!" and other debug messages
                engine.SetVariable("debug_file", OperatingSystem.IsWindows() ? "NUL" : "/dev/null");
                try
                {
                    using var pix = Pix.LoadFromMemory(item.Bitmap.PngBytes!);
                    using var page = engine.Process(pix);
                    var text = page.GetText()?.Trim();
                    results[item.ResultIndex] = string.IsNullOrWhiteSpace(text) ? null : text;
                }
                catch
                {
                    results[item.ResultIndex] = null;
                }
                await Task.CompletedTask; // Satisfy async signature
            });
    }
    catch (Exception ex)
    {
        WriteLine($"{fileName}: Tesseract OCR failed: {ex.Message}", ConsoleColor.Red);
        WriteLine($"{fileName}: Make sure Tesseract language data is available at: {tessdataPath}", ConsoleColor.Yellow);
        return string.Empty;
    }

    // Combine results in order
    var sb = new StringBuilder();
    var processedCount = 0;
    foreach (var text in results)
    {
        if (text is not null)
        {
            sb.AppendLine(text);
            processedCount++;
        }
    }

    WriteLine($"{fileName}: OCR completed, processed {processedCount} subtitle frames.", ConsoleColor.Gray);
    return sb.ToString();
}

static string GetOcrTextForDuration(List<BluRaySupParser.PcsData> pgsSubtitles, string fullOcrText, TimeSpan duration)
{
    // Count how many subtitles fall within the duration
    var count = pgsSubtitles.Count(s => s.StartTimeCode.TotalMilliseconds <= duration.TotalMilliseconds);
    
    // Split the full OCR text by lines and take approximately the right proportion
    var lines = fullOcrText.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries);
    var totalSubtitles = pgsSubtitles.Count;
    
    if (totalSubtitles == 0 || count == 0)
    {
        return string.Empty;
    }

    var linesToTake = (int)Math.Ceiling((double)lines.Length * count / totalSubtitles);
    return string.Join(Environment.NewLine, lines.Take(linesToTake));
}

async Task ProcessFileWithAudioTranscription(string filePath, int season, string fileName)
{
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
            // Save transcription to a file for debugging
            var transcriptionsDir = Path.Combine(Path.GetDirectoryName(filePath) ?? string.Empty, "transcriptions");
            Directory.CreateDirectory(transcriptionsDir);
            var transcriptionPath = Path.Combine(transcriptionsDir, Path.ChangeExtension(Path.GetFileName(filePath), ".txt"));
            await File.WriteAllTextAsync(transcriptionPath, transcription);

            // Step 3: Match transcription with subtitle files
            var bestMatch = FindBestMatchingSubtitle(filePath, showName, season, transcription, subtitles[season], chunk);

            if (bestMatch is not null)
            {
                // Step 4: Rename .mkv file
                RenameFile(showName, season, inputFolder, filePath, bestMatch);
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

static (bool HasTextSubs, bool HasImageSubs, string? ImageFormat) TryExtractEmbeddedSubtitles(string ffmpegPath, string inputFile, string outputTextSubtitle, string outputSupFile)
{
    var fileName = Path.GetFileName(inputFile);
    WriteLine($"{fileName}: Checking for embedded subtitles ...");

    // First, probe the file to find subtitle streams
    var (textStreamIndex, imageStreamIndex, imageFormat) = FindSubtitleStreamIndices(ffmpegPath, inputFile);

    var hasTextSubs = false;
    var hasImageSubs = false;

    // Try to extract text-based subtitles first (no codec copy needed, ffmpeg can convert to SRT)
    if (textStreamIndex is not null)
    {
        hasTextSubs = ExtractSubtitleStream(ffmpegPath, inputFile, outputTextSubtitle, textStreamIndex.Value, fileName, copyCodec: false);
    }

    // Extract image-based subtitles with codec copy (PGS/SUP cannot be re-encoded)
    if (imageStreamIndex is not null)
    {
        hasImageSubs = ExtractSubtitleStream(ffmpegPath, inputFile, outputSupFile, imageStreamIndex.Value, fileName, copyCodec: true);
    }

    if (!hasTextSubs && !hasImageSubs)
    {
        WriteLine($"{fileName}: No suitable subtitle stream found.");
    }

    return (hasTextSubs, hasImageSubs, imageFormat);
}

static bool ExtractSubtitleStream(string ffmpegPath, string inputFile, string outputFile, int streamIndex, string fileName, bool copyCodec = false)
{
    WriteLine($"{fileName}: Extracting embedded subtitle stream {streamIndex} to {Path.GetFileName(outputFile)} ...");

    // Use -c:s copy for image-based subtitles (SUP/PGS) to avoid re-encoding
    var codecArg = copyCodec ? "-c:s copy" : "";
    
    var process = new Process
    {
        StartInfo = new ProcessStartInfo
        {
            FileName = ffmpegPath,
            Arguments = $"-i \"{inputFile}\" -y -map 0:{streamIndex} {codecArg} \"{outputFile}\"",
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

    if (!process.WaitForExit(TimeSpan.FromSeconds(60)))
    {
        process.Kill();
        WriteLine($"{fileName}: Subtitle extraction timed out.", ConsoleColor.Yellow);
        return false;
    }

    if (process.ExitCode != 0)
    {
        WriteLine($"{fileName}: Failed to extract subtitles: {errorBuilder}", ConsoleColor.Yellow);
        return false;
    }

    // Verify the output file exists and has content
    if (!File.Exists(outputFile) || new FileInfo(outputFile).Length == 0)
    {
        WriteLine($"{fileName}: Extracted subtitle file is empty or doesn't exist.", ConsoleColor.Yellow);
        return false;
    }

    WriteLine($"{fileName}: Subtitle extraction complete.");
    return true;
}

static (int? TextStreamIndex, int? ImageStreamIndex, string? ImageFormat) FindSubtitleStreamIndices(string ffmpegPath, string inputFile)
{
    var fileName = Path.GetFileName(inputFile);

    // Use ffprobe (or ffmpeg with -i) to find subtitle streams
    // We prefer text-based subtitles (subrip/srt, ass, ssa) over image-based ones (dvdsub, pgs)
    var process = new Process
    {
        StartInfo = new ProcessStartInfo
        {
            FileName = ffmpegPath,
            Arguments = $"-i \"{inputFile}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        }
    };

    var errorBuilder = new StringBuilder();
    process.ErrorDataReceived += (sender, args) => errorBuilder.AppendLine(args.Data);

    process.Start();
    process.BeginErrorReadLine();
    process.WaitForExit(TimeSpan.FromSeconds(10));

    var output = errorBuilder.ToString();

    // Parse ffmpeg output to find subtitle streams
    // Example: Stream #0:2(eng): Subtitle: subrip
    // We prefer English subtitles and text-based formats
    var textBasedFormats = new[] { "subrip", "srt", "ass", "ssa", "webvtt", "mov_text" };
    var imagBasedFormats = new[] { "dvd", "pgs", "hdmv", "bitmap", "dvb", "xsub" };
    var preferredLanguages = new[] { "eng", "en" };

    int? bestTextStreamIndex = null;
    int bestTextScore = -1;
    int? bestImageStreamIndex = null;
    int bestImageScore = -1;
    string? bestImageFormat = null;

    foreach (var line in output.Split('\n'))
    {
        var streamMatch = SubtitleStreamRegex().Match(line);
        if (streamMatch.Success)
        {
            var streamIndex = int.Parse(streamMatch.Groups[1].Value);
            var language = streamMatch.Groups[2].Value.ToLowerInvariant();
            var format = streamMatch.Groups[3].Value.ToLowerInvariant();

            // Check if image-based subtitle format
            if (imagBasedFormats.Any(f => format.Contains(f)))
            {
                WriteLine($"{fileName}: Found image-based subtitle stream {streamIndex} ({format}).", ConsoleColor.Gray);
                
                // Calculate a score based on language preferences
                var score = 0;
                if (preferredLanguages.Any(l => language.Contains(l)))
                {
                    score += 5;
                }

                if (score > bestImageScore)
                {
                    bestImageScore = score;
                    bestImageStreamIndex = streamIndex;
                    bestImageFormat = format;
                }
                continue;
            }

            // Text-based subtitle
            // Calculate a score based on format and language preferences
            var textScore = 0;
            if (textBasedFormats.Any(f => format.Contains(f)))
            {
                textScore += 10;
            }
            if (preferredLanguages.Any(l => language.Contains(l)))
            {
                textScore += 5;
            }

            if (textScore > bestTextScore)
            {
                bestTextScore = textScore;
                bestTextStreamIndex = streamIndex;
            }
        }
    }

    if (bestTextStreamIndex is null && bestImageStreamIndex is null)
    {
        WriteLine($"{fileName}: No subtitle streams found.", ConsoleColor.Yellow);
    }

    return (bestTextStreamIndex, bestImageStreamIndex, bestImageFormat);
}

// TODO: Pass in the StringBuilder and pool them at the call site
async Task<string> TranscribeAudio(string audioPath, int season)
{
    var fileName = Path.GetFileName(audioPath);
    WriteLine($"{fileName}: Transcribing audio from {fileName} ...");

    var sb = new StringBuilder();

    var factory = await GetWhisperFactory();
    var whisperBuilder = CreateWhisperBuilder(factory, showName, season);
    await using var processor = whisperBuilder.Build();

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

    var downloader = new WhisperGgmlDownloader(new());
    using var modelStream = await downloader.GetGgmlModelAsync(ggmlType);
    using var fileStream = File.OpenWrite(filePath);
    await modelStream.CopyToAsync(fileStream);
}

static async Task EnsureTessdataExists(string tessdataPath)
{
    Directory.CreateDirectory(tessdataPath);
    
    var engTrainedData = Path.Join(tessdataPath, "eng.traineddata");
    if (File.Exists(engTrainedData))
    {
        return;
    }

    WriteLine("Downloading Tesseract English language data for OCR...", ConsoleColor.Gray);
    
    // Download eng.traineddata from tessdata_fast repository
    const string tessdataUrl = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata";
    
    using var httpClient = new HttpClient();
    httpClient.Timeout = TimeSpan.FromMinutes(5);
    
    try
    {
        var response = await httpClient.GetAsync(tessdataUrl);
        response.EnsureSuccessStatusCode();
        
        await using var fileStream = File.Create(engTrainedData);
        await response.Content.CopyToAsync(fileStream);
        
        WriteLine("Tesseract language data downloaded successfully.", ConsoleColor.Green);
    }
    catch (Exception ex)
    {
        WriteLine($"Failed to download Tesseract language data: {ex.Message}", ConsoleColor.Red);
        WriteLine($"Please manually download eng.traineddata from {tessdataUrl} to {tessdataPath}", ConsoleColor.Yellow);
    }
}

static string? FindBestMatchingSubtitle(string filePath, string showName, int season, string transcription, List<(string, string[])> subtitles, int chunk, bool isOcrText = false)
{
    WriteLine($"{Path.GetFileName(filePath)}: Finding best matching subtitle");

    // Normalize the transcription/OCR text
    var normalizedTranscription = NormalizeTextForMatching(transcription);

    string? bestMatch = null;
    double highestSimilarity = 0;

    foreach (var subtitle in subtitles)
    {
        var fileName = subtitle.Item1;
        var subtitleContent = string.Join(Environment.NewLine, subtitle.Item2.Take(chunk));
        var normalizedSubtitle = NormalizeTextForMatching(subtitleContent);

        var similarity = CalculateCosineSimilarity(normalizedTranscription, normalizedSubtitle);
        if (similarity > highestSimilarity)
        {
            highestSimilarity = similarity;
            bestMatch = fileName;
        }
    }

    // Use a lower threshold for OCR text since it inherently has more errors
    var threshold = isOcrText ? 0.5 : 0.8;
    
    if (highestSimilarity < threshold)
    {
        WriteLine($"{Path.GetFileName(filePath)}: No matching subtitle above {threshold:P0} found (highest was {Path.GetFileName(bestMatch)} with similarity of {highestSimilarity:P2}).", ConsoleColor.Yellow);
        return null;
    }

    WriteLine($"{Path.GetFileName(filePath)}: Match found! Best match is {Path.GetFileName(bestMatch)} with similarity of {highestSimilarity:P2}.", ConsoleColor.Green);
    return bestMatch;
}

static string NormalizeTextForMatching(string text)
{
    if (string.IsNullOrWhiteSpace(text))
    {
        return string.Empty;
    }

    var sb = new StringBuilder(text.Length);
    
    foreach (var c in text)
    {
        if (char.IsLetterOrDigit(c))
        {
            sb.Append(char.ToLowerInvariant(c));
        }
        else if (char.IsWhiteSpace(c) || c == '\n' || c == '\r')
        {
            // Normalize all whitespace to single space
            if (sb.Length > 0 && sb[^1] != ' ')
            {
                sb.Append(' ');
            }
        }
        // Skip punctuation and other characters
    }

    return sb.ToString().Trim();
}

static Dictionary<int, List<(string, string[])>> LoadSubtitles(string showName, string subtitlesFolder, TimeSpan duration, int chunkCount)
{
    // TODO: Change to use globbing via Microsoft.Extensions.FileSystemGlobbing
    var srtFiles = Directory.GetFiles(subtitlesFolder, "*.srt")
        .Select(file =>
        {
            var seasonParsed = int.TryParse(ExtractSeasonEpisode(file, justSeason: true), out var season);
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
                
                // Skip very short words (likely OCR artifacts or noise)
                if (key.Length < 2)
                {
                    continue;
                }
                
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

static string? ExtractSeasonEpisode(string fileName, bool justSeason = false)
{
    var match = SeasonEpisodeRegex().Match(Path.GetFileNameWithoutExtension(fileName));
    return match.Success ? match.Groups[justSeason ? 1 : 0].Value : null;
}

void RenameFile(string showName, int season, string inputFolder, string currentFilePath, string bestMatch)
{
    var seasonEpisode = ExtractSeasonEpisode(bestMatch);
    if (!string.IsNullOrEmpty(seasonEpisode))
    {
        var newFileName = $"{showName} - {seasonEpisode}.mkv";
        var newFilePath = Path.Join(inputFolder, $"Season {season:D2}", newFileName);

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
    var modelFileName = $"ggml-{Enum.GetName(ggmlType)!.ToLowerInvariant()}.bin";
    var modelFilePath = Path.Join(cacheDir, "whisper", modelFileName);

    if (!File.Exists(modelFilePath))
    {
        await DownloadModel(modelFilePath, ggmlType);
    }

    var whisperFactory = WhisperFactory.FromPath(modelFilePath);

    return whisperFactory;
}

static WhisperProcessorBuilder CreateWhisperBuilder(WhisperFactory whisperFactory, string showName, int season)
{
    var whisperBuilder = whisperFactory.CreateBuilder()
        .WithLanguage("en")
        .WithPrompt($"This is a fragment of the audio track from an episode of the TV series '{showName}' in season {season}")
        .WithThreads(4);

    return whisperBuilder;
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

    // Matches ffmpeg stream output like: Stream #0:2(eng): Subtitle: subrip
    // Group 1: stream index, Group 2: language (optional), Group 3: subtitle format
    [GeneratedRegex(@"Stream #0:(\d+)(?:\(([a-z]{2,3})\))?:\s*Subtitle:\s*(\w+)", RegexOptions.IgnoreCase, "en-US")]
    public static partial Regex SubtitleStreamRegex();

    [GeneratedRegex(@"<[^>]+>")]
    public static partial Regex HtmlTags();
}
