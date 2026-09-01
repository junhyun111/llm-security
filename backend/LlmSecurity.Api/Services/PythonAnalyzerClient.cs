using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;

namespace LlmSecurity.Api.Services;

public class PythonAnalyzerClient
{
    private readonly HttpClient _http;
    private readonly IHttpContextAccessor _httpContextAccessor;

    public PythonAnalyzerClient(
        HttpClient http,
        IHttpContextAccessor httpContextAccessor)
    {
        _http = http;
        _httpContextAccessor = httpContextAccessor;
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

    public async Task<string> GetRuntimeMetadataJsonAsync(
        CancellationToken cancellationToken = default)
    {
        using var response = await _http.GetAsync("/api/runtime", cancellationToken);
        await EnsureSuccess(response, cancellationToken);
        return await response.Content.ReadAsStringAsync(cancellationToken);
    }

    public async Task<PythonJobDto> CreateJobAsync(
        string projectName,
        IReadOnlyList<IFormFile> files,
        IReadOnlyList<string> relativePaths,
        CancellationToken cancellationToken = default)
    {
        var requestOptions = await ReadRequestOptionsAsync(cancellationToken);

        using var content = new MultipartFormDataContent();
        AddRunOptions(
            content,
            projectName,
            requestOptions.Sensitivity,
            requestOptions.Model,
            requestOptions.ApiKey);

        for (var i = 0; i < files.Count; i++)
        {
            content.Add(
                new StringContent(relativePaths[i], Encoding.UTF8),
                "relative_paths");

            var file = files[i];
            var streamContent = new StreamContent(file.OpenReadStream());

            if (!string.IsNullOrWhiteSpace(file.ContentType))
            {
                streamContent.Headers.ContentType =
                    new System.Net.Http.Headers.MediaTypeHeaderValue(file.ContentType);
            }

            content.Add(streamContent, "files", file.FileName);
        }

        return await SendCreateJobAsync(content, cancellationToken);
    }

    public async Task<PythonJobDto> CreateJobFromLocalPathAsync(
        string projectName,
        IReadOnlyList<LocalProjectFile> files,
        double sensitivity,
        string? model,
        string? apiKey,
        CancellationToken cancellationToken = default)
    {
        if (files.Count == 0)
            throw new ArgumentException("No local project files were supplied.", nameof(files));

        ValidateRunOptions(ref sensitivity, ref model, ref apiKey);

        using var content = new MultipartFormDataContent();
        AddRunOptions(
            content,
            projectName,
            sensitivity,
            model,
            apiKey);

        foreach (var file in files)
        {
            content.Add(
                new StringContent(file.RelativePath, Encoding.UTF8),
                "relative_paths");

            var stream = new FileStream(
                file.FullPath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 64 * 1024,
                options: FileOptions.Asynchronous | FileOptions.SequentialScan);

            var streamContent = new StreamContent(stream);
            content.Add(streamContent, "files", file.FileName);
        }

        return await SendCreateJobAsync(content, cancellationToken);
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
            await EnsureSuccess(response, cancellationToken);

        return response;
    }

    private async Task<PythonJobDto> SendCreateJobAsync(
        MultipartFormDataContent content,
        CancellationToken cancellationToken)
    {
        using var response = await _http.PostAsync(
            "/api/jobs",
            content,
            cancellationToken);

        await EnsureSuccess(response, cancellationToken);

        return (await response.Content.ReadFromJsonAsync<PythonJobDto>(
            cancellationToken: cancellationToken))!;
    }

    private static void AddRunOptions(
        MultipartFormDataContent content,
        string projectName,
        double sensitivity,
        string? model,
        string? apiKey)
    {
        content.Add(
            new StringContent(projectName, Encoding.UTF8),
            "project_name");

        content.Add(
            new StringContent(
                sensitivity.ToString(CultureInfo.InvariantCulture),
                Encoding.UTF8),
            "sensitivity");

        if (!string.IsNullOrWhiteSpace(model))
        {
            content.Add(
                new StringContent(model, Encoding.UTF8),
                "model");
        }

        if (!string.IsNullOrWhiteSpace(apiKey))
        {
            content.Add(
                new StringContent(apiKey, Encoding.UTF8),
                "api_key");
        }
    }

    private async Task<AnalysisRequestOptions> ReadRequestOptionsAsync(
        CancellationToken cancellationToken)
    {
        var sensitivity = 0.5;
        string? model = null;
        string? apiKey = null;

        var request = _httpContextAccessor.HttpContext?.Request;
        if (request?.HasFormContentType != true)
            return new AnalysisRequestOptions(sensitivity, model, apiKey);

        var form = await request.ReadFormAsync(cancellationToken);

        var rawSensitivity = form["sensitivity"].ToString().Trim();
        if (!string.IsNullOrWhiteSpace(rawSensitivity))
        {
            if (!double.TryParse(
                    rawSensitivity,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out sensitivity))
            {
                throw new AnalyzerApiException(
                    HttpStatusCode.BadRequest,
                    "민감도는 0.0에서 1.0 사이여야 합니다.");
            }
        }

        model = NormalizeOptional(form["model"].ToString());
        apiKey = NormalizeOptional(form["api_key"].ToString());

        ValidateRunOptions(ref sensitivity, ref model, ref apiKey);

        return new AnalysisRequestOptions(sensitivity, model, apiKey);
    }

    private static void ValidateRunOptions(
        ref double sensitivity,
        ref string? model,
        ref string? apiKey)
    {
        if (sensitivity < 0.0 || sensitivity > 1.0)
        {
            throw new AnalyzerApiException(
                HttpStatusCode.BadRequest,
                "민감도는 0.0에서 1.0 사이여야 합니다.");
        }

        model = NormalizeOptional(model);
        if (model is { Length: > 200 } ||
            model?.Contains('\r') == true ||
            model?.Contains('\n') == true)
        {
            throw new AnalyzerApiException(
                HttpStatusCode.BadRequest,
                "유효하지 않은 모델 ID입니다.");
        }

        apiKey = NormalizeOptional(apiKey);
        if (apiKey is { Length: > 512 } ||
            apiKey?.Any(char.IsWhiteSpace) == true)
        {
            throw new AnalyzerApiException(
                HttpStatusCode.BadRequest,
                "유효하지 않은 OpenRouter API Key 형식입니다.");
        }
    }

    private static string? NormalizeOptional(string? value)
    {
        var normalized = value?.Trim();
        return string.IsNullOrWhiteSpace(normalized) ? null : normalized;
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
            {
                message = detail.ValueKind == JsonValueKind.String
                    ? detail.GetString() ?? body
                    : detail.GetRawText();
            }
            else if (doc.RootElement.TryGetProperty("message", out var apiMessage))
            {
                message = apiMessage.GetString() ?? body;
            }
        }
        catch (JsonException)
        {
            // 원문을 그대로 사용한다.
        }

        throw new AnalyzerApiException(response.StatusCode, message);
    }

    private sealed record AnalysisRequestOptions(
        double Sensitivity,
        string? Model,
        string? ApiKey);
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
