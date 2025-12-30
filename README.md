# OpenSubtitleDownloader & MkvFileMatcher

Two scrappy utilities in C# that can be used to help organize rips of TV series from DVD/Blu-ray boxsets. Many TV series boxsets don't have the episodes in the correct order on the discs, so when they're ripped using a tool like MakeMKV, you end up with a bunch of video files without any idea which one is which episode. These utilities help you automate the process of identifying which file is which episode by matching subtitles extracted from the files with the best match from  downloaded reference subtitles.

As noted above, this utilities are currently in a pretty scrappy state. The only way to use them is to run them from source and edit the code to match the shows you're working with. This of course can be updated in good time to make them more user-friendly but for now, they work for me which is good enough 😁.

- **OpenSubtitleDownloader**:

   A utility to download subtitle files for TV shows from [OpenSubtitles](https://opensubtitles.com).

   You'll need an OpenSubtitles account and API key which can be created on the [API Consumers page](https://www.opensubtitles.com/en/consumers).

- **MkvFileMatcher**:

   A utility to process MKV files and rename them to match the season and episode they're for. Requires reference subtitles to first be downloaded using OpenSubtitleDownloader. Extracts subtitles from the files using `ffmpeg` (must be on your path) and matches them using cosine similarity to the matching episode from the downloaded subtitles. If the MKV contains embedded subtitles, it will use them (either text-based or image-based), otherwise it will fall back to extracting the audio track and transcribing it to text using Whisper (works best if you have a nice GPU).

## Usage

These tools assume you have already ripped your TV series discs to MKV files using a tool like [MakeMKV](https://www.makemkv.com/). The files should be placed in folders named for the seasons they're from. **This is very important**. Make sure you have folder names like `Season 01`, `Season 02`, etc., and that each contains only the MKV files for the episodes in that season. Each file should have a unique name, e.g. `Brooklyn Nine-Nine - s01d01t00.mkv` (`s01` indicates season 1, `d01` indicates disc 1, `t00` indicates title 0 from the disc). During processing, these files will be renamed in place by the utility to match the episode, e.g. `Brooklyn Nine-Nine - S01E01.mkv`.

### Download series subtitles (REQUIRED!)

1. Clone this repo
2. Create an account and/or login at https://www.opensubtitles.com/
3. Register an OpenSubtitles API consumer on the https://www.opensubtitles.com/en/consumers page
4. In the `./OpenSubtitleDownloader` directory, set the following required configuration values using user secrets, e.g. `dotnet user-secrets set username myusername`:
   - `ProductName`: The product name you entered for your API consumer on the https://www.opensubtitles.com/en/consumers page
   - `ProductVersion`: A version to pass to the OpenSubtitles API, e.g. `v1.0`
   - `ApiKey`: The API key listed on the https://www.opensubtitles.com/en/consumers page for the registered consumer
   - `username`: Your OpenSubtitles username
   - `password`: Your OpenSubtitles password
5. Verify all the required secrets are configured by running `dotnet user-secrets list` in the `./OpenSubtitleDownloader` directory, e.g.:
    ```shell
    $ dotnet user-secrets list
    username = DamianEdwards
    ProductVersion = v0.1
    ProductName = damianmkvmatching
    password = *************
    ApiKey = **********************************
    ```
6. Open `./OpenSubtitleDownloader/Program.cs` and locate the code block that sets the show name and year for the search, and update it to match the show you're processing files for. Note you can manually search for it first at https://www.opensubtitles.com to ensure you get the query name and year correct:
    ```csharp
    var featureSearch = new NewFeatureSearch
    {
        Query = "Brooklyn Nine-Nine",
        Year = 2013
    };
    ```
7. Run the app to download the subtitles files: `dotnet run`
8. The files will be downloaded and saved in the cache directory in your user profile: `~/.subtitlr`

### Match MVK files to episodes

1. Open `./MkvFileMatcher/Program.cs` and locate the code block early in the file that sets the show name and season number that you want to process, along with the path to the directory containing the show MKV files, and update it accordingly:
   ```csharp
   var showName = "Brooklyn Nine-Nine";
   int? specificSeason = 8; // Leave as null to process all seasons
   int? episode = null; // Used for debugging specific episode
   var inputFolder = @$"G:\Video\{showName}";
   ```
2. Run the app to process the MKV files: `dotnet run`
3. If any shows can't be matched or you see messages indicating that a match was found but the filename for that episode already existed (indicating you likely got some false positive matches), you'll need to do some clean-up. If so, read on.

#### Manually matching MKV files to episodes by subtitle referencing
1. To manually clean-up mismatches or files with no found matches, open the directory containing all the downloaded reference subtitle files in an editor like VS Code. It will be something like `~/.subtitlr/<show-name>`. Now you can easily search for text amongst all these files.
2. Open the MKV file you want to match in a player like [VLC Media Player](https://www.videolan.org/) and watch it until you hear a short phrase you think is a good candidate to search for. Turn on subtitles to easily see what's being said, and try to pick a single word or very short phrase that is likely to be unique to this episode.
3. In your editor, search for the phrase you chose and see if a match is found in any of the downloaded subtitle files. You might need to try including punctuation like commas or omitting leading/trailing words. This is why a short phrase works best (single word if you can!).
4. Look at the matches and determine if the words around the found match line up with the audio in the episode you were watching. If they do, the name of the subtitle file will indicate which episode it is!