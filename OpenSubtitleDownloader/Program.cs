using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading.RateLimiting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using MovieCollection.OpenSubtitles;
using MovieCollection.OpenSubtitles.Models;
using OpenSubtitleDownloader;

var builder = new HostApplicationBuilder(args);
builder.Configuration.AddUserSecrets<Program>();
builder.Logging.SetMinimumLevel(LogLevel.Debug);

var host = builder.Build();

var logger = host.Services.GetRequiredService<ILoggerFactory>()
    .CreateLogger(builder.Environment.ApplicationName);

using var httpClient = new HttpClient(handler: new RateLimitedHttpHandler(
        limiter: new TokenBucketRateLimiter(new()
        {
            TokenLimit = 5,
            QueueLimit = 3,
            ReplenishmentPeriod = TimeSpan.FromSeconds(1),
            TokensPerPeriod = 5,
            AutoReplenishment = true
        })));

var username = builder.Configuration["username"] ?? throw new InvalidOperationException("username not found in configuration");
var password = builder.Configuration["password"] ?? throw new InvalidOperationException("password not found in configuration");

var options = new OpenSubtitlesOptions
{
    ApiKey = builder.Configuration["ApiKey"] ?? throw new InvalidOperationException("ApiKey not found in configuration."),
    ProductInformation = new ProductHeaderValue(
        builder.Configuration["ProductName"] ?? throw new InvalidOperationException("ProductName not found in configuration"),
        builder.Configuration["ProductVersion"] ?? throw new InvalidOperationException("ProductVersion not found in configuration")),
};

var service = new OpenSubtitlesService(httpClient, options);

var login = await service.LoginAsync(new() { Username = username, Password = password });

var featureSearch = new NewFeatureSearch
{
    Query = "The Big Bang Theory"
};

var featureResult = await service.SearchFeaturesAsync(featureSearch);
if (featureResult.Data is null || featureResult.Data.Count == 0)
{
    throw new InvalidOperationException($"No features found for search query '{featureSearch.Query}'");
}

var showDetails = featureResult.Data.FirstOrDefault(r => r?.Attributes?.OriginalTitle == featureSearch.Query && r?.Attributes.SeasonNumber == 0)?.Attributes;

if (showDetails is null)
{
    throw new InvalidOperationException($"No feature result found with title '{featureSearch.Query}'");
}

logger.LogInformation("Feature '{ShowTitle}' with {SeasonCount} seasons found!", showDetails.OriginalTitle, showDetails.SeasonsCount);

Dictionary<int, List<SubtitleFile>>? subtitleFiles = null;

var cacheDirPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), $".subtitlr");
if (!Directory.Exists(cacheDirPath))
{
    Directory.CreateDirectory(cacheDirPath);
}
var cacheFilePath = Path.Combine(cacheDirPath, $"{showDetails.OriginalTitle!}.json");

if (!Path.Exists(cacheFilePath))
{
    logger.LogDebug("No cache file for {ShowTitle} found at '{CachePath}', proceeding to search for subtitles", showDetails.OriginalTitle, cacheFilePath);

    var search = new NewSubtitleSearch
    {
        ParentFeatureId = showDetails.FeatureId,
        Query = showDetails.OriginalTitle,
        Languages = ["en"]
    };

    var subtitleResults = new List<Subtitle>();

    var page = 1;
    while (true)
    {
        search.Page = page;

        logger.LogInformation("Searching for subtitles for '{ShowName}', page {Page}", showDetails.OriginalTitle, search.Page);

        var response = await service.SearchSubtitlesAsync(search);

        if (response.TotalCount == 0 || response?.Data is null || response.Data.Count == 0)
        {
            logger.LogInformation("No search results found");
            break;
        }

        foreach (var result in response.Data)
        {
            if (result?.Attributes is not null)
            {
                subtitleResults.Add(result.Attributes);
            }
        }

        if (page == response.TotalPages)
        {
            break;
        }

        page++;
    }

    logger.LogDebug("Found {ResultCount} subtitles", subtitleResults.Count);

    logger.LogDebug("Filtering results to most downloaded files");

    subtitleFiles = subtitleResults
        .Where(s => s.FeatureDetails is { } features
                    && features.FeatureId is { }
                    && features.ParentTitle is { }
                    && features.SeasonNumber is { }
                    && features.EpisodeNumber is { }
                    && s.Files is { Count: > 0 })
        .OrderBy(s => s.FeatureDetails!.SeasonNumber!.Value)
        .GroupBy(s => s.FeatureDetails!.SeasonNumber!.Value)
        .ToDictionary(s => /* Season */ s.Key, s => s.OrderBy(s => s.FeatureDetails!.EpisodeNumber!.Value)
                                                     .GroupBy(s => s.FeatureDetails!.EpisodeNumber!.Value)
                                                     .ToDictionary(e => /* Episode */ e.Key, e => e.OrderByDescending(f => Math.Max(f.DownloadCount, f.NewDownloadCount))
                                                                                                   .First())
                                                     .Values.Select(s => new SubtitleFile(
                                                                                showDetails.OriginalTitle!,
                                                                                s.FeatureDetails!.SeasonNumber!.Value,
                                                                                s.FeatureDetails!.EpisodeNumber!.Value,
                                                                                s.Files!.First()))
                                                            .ToList());

    using var cacheFile = File.Open(cacheFilePath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
    await JsonSerializer.SerializeAsync(cacheFile, subtitleFiles);

    logger.LogDebug("Cache file for {ShowTitle} written to '{CachePath}'", showDetails.OriginalTitle, cacheFilePath);
}
else
{
    logger.LogDebug("Reading cache file for {ShowTitle} from {CachePath}", showDetails.OriginalTitle, cacheFilePath);
    using var cacheFile = File.OpenRead(cacheFilePath);
    subtitleFiles = JsonSerializer.Deserialize<Dictionary<int, List<SubtitleFile>>>(cacheFile);
}

if (subtitleFiles is null)
{
    throw new InvalidOperationException("This shouldn't be null");
}

var totalSubtitleCount = subtitleFiles.Sum(s => s.Value.Count);
logger.LogInformation("Found subtitles for {TotalEpisodeCount} episodes in {SeasonCount} seasons.", totalSubtitleCount, subtitleFiles.Count);

// Download files
var downloadDirPath = Path.Combine(cacheDirPath, showDetails.OriginalTitle!);

if (!Directory.Exists(downloadDirPath))
{
    Directory.CreateDirectory(downloadDirPath);
}

var quotaExceeded = false;
foreach (var season in subtitleFiles.Keys)
{
    var targetSeasonDirPath = Path.Combine(downloadDirPath, $"Season {season:D2}");
    if (!Directory.Exists(targetSeasonDirPath))
    {
        Directory.CreateDirectory(targetSeasonDirPath);
    }

    var seasonFiles = subtitleFiles[season];
    foreach (var file in seasonFiles)
    {
        var targetFileName = $"{file.ShowName} - S{file.SeasonNumber:D2}E{file.EpisodeNumber:D2}.srt";
        var targetFilePath = Path.Combine(targetSeasonDirPath, targetFileName);

        if (Path.Exists(targetFilePath))
        {
            logger.LogInformation("Subtitle file '{FilePath}' for show '{ShowTitle}', season {SeasonNumber}, episode {EpisodeNumber} already exists! Skipping download.", downloadDirPath, file.ShowName, file.SeasonNumber, file.EpisodeNumber);
            continue;
        }

        logger.LogInformation("Downloading subtitle file for '{ShowTitle}', season {SeasonNumber}, episode {EpisodeNumber}", file.ShowName, file.SeasonNumber, file.EpisodeNumber);
        var download = await service.GetSubtitleForDownloadAsync(new() { FileId = file.File.FileId, SubFormat = "srt" }, login.Token);
        if (download.Link is not null)
        {
            logger.LogDebug("Downloading file from {DownloadUrl}", download.Link);

            var downloadRequest = new HttpRequestMessage(HttpMethod.Get, download.Link);
            downloadRequest.Headers.Authorization = new("Bearer", login.Token);
            downloadRequest.Headers.Add("Api-Key", options.ApiKey);
            downloadRequest.Headers.Add("User-Agent", options.ProductInformation.ToString());

            var downloadResponse = await httpClient.SendAsync(downloadRequest);
            downloadResponse.EnsureSuccessStatusCode();
            using var downloadStream = await downloadResponse.Content.ReadAsStreamAsync();
            await using var targetFileStream = File.OpenWrite(targetFilePath);
            await downloadStream.CopyToAsync(targetFileStream);

            logger.LogDebug("Finished downloading file '{FilePath}' from {DownloadUrl}", targetFilePath, download.Link);
        }
        else if (download.Remaining <= 0)
        {
            quotaExceeded = true;
            logger.LogWarning("Download quota exhausted: {Message}", download.Message);
            break;
        }
        else
        {
            logger.LogWarning("Download link was not returned: {Message}", download.Message);
            break;
        }
    }

    if (quotaExceeded)
    {
        break;
    }
}

logger.LogInformation("Downloads complete! Press ENTER to exit.");

Console.ReadLine();

record SubtitleFile(string ShowName, int SeasonNumber, int EpisodeNumber, MovieCollection.OpenSubtitles.Models.SubtitleFile File);
