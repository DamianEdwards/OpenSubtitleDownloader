using System.Threading.RateLimiting;

namespace OpenSubtitleDownloader;
internal sealed class RateLimitedHttpHandler(RateLimiter limiter)
    : DelegatingHandler(new HttpClientHandler()), IAsyncDisposable
{
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var acquired = false;

        while (!acquired)
        {
            using RateLimitLease lease = await limiter.AcquireAsync(permitCount: 1, cancellationToken);
            acquired = lease.IsAcquired;
            if (acquired)
            {
                break;
            }

            await Task.Delay(100, cancellationToken);
        }

        return await base.SendAsync(request, cancellationToken);
    }

    async ValueTask IAsyncDisposable.DisposeAsync()
    {
        await limiter.DisposeAsync().ConfigureAwait(false);

        Dispose(disposing: false);
        GC.SuppressFinalize(this);
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);

        if (disposing)
        {
            limiter.Dispose();
        }
    }
}
