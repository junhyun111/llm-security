using System.Net;
using System.Net.Http.Json;
using System.Text.Json;

namespace LlmSecurity.Api.Services;

public class PythonAnalyzerClient
{
    private readonly HttpClient _http;

    public PythonAnalyzerClient(HttpClient http)
    {
        _http = http;
    }

    public async Task<bool> IsHealthyAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            using var response = await _http.GetAsync("/api/health", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task<PythonJobDto> CreateJobAsync(
        string projectName,
        IReadOnlyList<IFormFile> files,
        IReadOnlyList<string> relativePaths,
        CancellationToken cancellationToken = default)
    {
        using var content = new MultipartFormDataContent();
        content.Add(new StringContent(projectName), "project_name");

        for (var i = 0; i < files.Count; i++)
        {
            content.Add(new StringContent(relativePaths[i]), "relative_paths");

            var file = files[i];
            var streamContent = new StreamContent(file.OpenReadStream());
            if (!string.IsNullOrWhiteSpace(file.ContentType))
                streamContent.Headers.ContentType =
                    new System.Net.Http.Headers.MediaTypeHeaderValue(file.ContentType);

            content.Add(streamContent, "files", file.FileName);
        }

        using var response = await _http.PostAsync("/api/jobs", content, cancellationToken);
        await EnsureSuccess(response, cancellationToken);

        return (await response.Content.ReadFromJsonAsync<PythonJobDto>(
            cancellationToken: cancellationToken))!;
    }

    public async Task<PythonJobDto> GetJobAsync(
        string analyzerJobId,
        CancellationToken cancellationToken = default)
    {
        using var response = await _http.GetAsync(
            $"/api/jobs/{Uri.EscapeDataString(analyzerJobId)}",
            cancellationToken);

        await EnsureSuccess(response, cancellationToken);

        return (await response.Content.ReadFromJsonAsync<PythonJobDto>(
            cancellationToken: cancellationToken))!;
    }

    public async Task<string> GetAnalysisJsonAsync(
        string analyzerJobId,
        CancellationToken cancellationToken = default)
    {
        using var response = await _http.GetAsync(
            $"/api/jobs/{Uri.EscapeDataString(analyzerJobId)}/analysis",
            cancellationToken);

        await EnsureSuccess(response, cancellationToken);
        return await response.Content.ReadAsStringAsync(cancellationToken);
    }

    public async Task<string> ProposePatchAsync(
        string analyzerJobId,
        IReadOnlyList<string> findingIds,
        CancellationToken cancellationToken = default)
    {
        using var response = await _http.PostAsJsonAsync(
            $"/api/jobs/{Uri.EscapeDataString(analyzerJobId)}/patches/proposal",
            new { finding_ids = findingIds },
            cancellationToken);

        await EnsureSuccess(response, cancellationToken);
        return await response.Content.ReadAsStringAsync(cancellationToken);
    }

    public async Task<string> PatchActionAsync(
        string analyzerJobId,
        string patchId,
        string action,
        CancellationToken cancellationToken = default)
    {
        if (action is not ("approve" or "reject"))
            throw new ArgumentException("Unsupported patch action.", nameof(action));

        using var response = await _http.PostAsync(
            $"/api/jobs/{Uri.EscapeDataString(analyzerJobId)}/patches/{Uri.EscapeDataString(patchId)}/{action}",
            null,
            cancellationToken);

        await EnsureSuccess(response, cancellationToken);
        return await response.Content.ReadAsStringAsync(cancellationToken);
    }

    public async Task<HttpResponseMessage> DownloadAsync(
        string analyzerJobId,
        CancellationToken cancellationToken = default)
    {
        var response = await _http.GetAsync(
            $"/api/jobs/{Uri.EscapeDataString(analyzerJobId)}/download",
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            await EnsureSuccess(response, cancellationToken);
        }

        return response;
    }

    private static async Task EnsureSuccess(
        HttpResponseMessage response,
        CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode)
            return;

        var body = await response.Content.ReadAsStringAsync(cancellationToken);
        var message = body;

        try
        {
            using var doc = JsonDocument.Parse(body);
            if (doc.RootElement.TryGetProperty("detail", out var detail))
                message = detail.ValueKind == JsonValueKind.String
                    ? detail.GetString() ?? body
                    : detail.GetRawText();
        }
        catch (JsonException)
        {
            // 원문을 그대로 사용한다.
        }

        throw new AnalyzerApiException(response.StatusCode, message);
    }
}

public sealed class AnalyzerApiException : Exception
{
    public HttpStatusCode StatusCode { get; }

    public AnalyzerApiException(HttpStatusCode statusCode, string message)
        : base(message)
    {
        StatusCode = statusCode;
    }
}
