using System.Net;
using LlmSecurity.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace LlmSecurity.Api.Controllers;

/// <summary>
/// 개발 중 모델/runtime 검증을 위한 로컬 경로 분석 API.
/// 운영 환경에서는 자동으로 비활성화된다.
/// </summary>
[ApiController]
[Route("api/dev/runtime")]
public sealed class RuntimeDebugController : ControllerBase
{
    private readonly IWebHostEnvironment _environment;
    private readonly IConfiguration _configuration;
    private readonly PythonAnalyzerClient _analyzer;
    private readonly LocalProjectFileCollector _collector;

    public RuntimeDebugController(
        IWebHostEnvironment environment,
        IConfiguration configuration,
        PythonAnalyzerClient analyzer,
        LocalProjectFileCollector collector)
    {
        _environment = environment;
        _configuration = configuration;
        _analyzer = analyzer;
        _collector = collector;
    }

    [HttpGet("status")]
    public async Task<IActionResult> Status(CancellationToken cancellationToken)
    {
        var guard = GuardDebugEndpoint();
        if (guard is not null)
            return guard;

        var allowedRoot = ResolveAllowedRoot();

        return Ok(new
        {
            enabled = true,
            environment = _environment.EnvironmentName,
            localhostOnly = true,
            allowedProjectRoot = allowedRoot,
            runtimeHealthy = await _analyzer.IsHealthyAsync(cancellationToken)
        });
    }

    [HttpPost("analyze-path")]
    public async Task<IActionResult> AnalyzePath(
        [FromBody] RuntimePathAnalysisRequest request,
        CancellationToken cancellationToken)
    {
        var guard = GuardDebugEndpoint();
        if (guard is not null)
            return guard;

        if (request.Sensitivity < 0.0 || request.Sensitivity > 1.0)
        {
            return BadRequest(new
            {
                message = "민감도는 0.0에서 1.0 사이여야 합니다."
            });
        }

        try
        {
            var snapshot = _collector.Collect(
                request.ProjectPath,
                ResolveAllowedRoot(),
                maxFiles: _configuration.GetValue<int?>(
                    "RuntimeDebug:MaxFiles") ?? 500,
                maxTotalBytes: Megabytes(
                    _configuration.GetValue<int?>(
                        "RuntimeDebug:MaxTotalMb") ?? 100),
                maxSingleFileBytes: Megabytes(
                    _configuration.GetValue<int?>(
                        "RuntimeDebug:MaxSingleFileMb") ?? 10));

            var projectName = string.IsNullOrWhiteSpace(request.ProjectName)
                ? snapshot.ProjectName
                : request.ProjectName.Trim();

            var remote = await _analyzer.CreateJobFromLocalPathAsync(
                projectName,
                snapshot.Files,
                request.Sensitivity,
                request.Model,
                request.ApiKey,
                cancellationToken);

            return Accepted(new
            {
                jobId = remote.JobId,
                projectName = remote.ProjectName,
                status = remote.Status,
                progress = remote.Progress,
                message = remote.Message,
                fileCount = remote.FileCount,
                sourceFileCount = remote.SourceFileCount,
                localProjectPath = snapshot.RootPath,
                uploadedBytes = snapshot.TotalBytes,
                statusEndpoint = $"/api/dev/runtime/jobs/{remote.JobId}",
                analysisEndpoint = $"/api/dev/runtime/jobs/{remote.JobId}/analysis"
            });
        }
        catch (AnalyzerApiException ex)
        {
            return StatusCode((int)ex.StatusCode, new { message = ex.Message });
        }
        catch (UnauthorizedAccessException ex)
        {
            return StatusCode(
                StatusCodes.Status403Forbidden,
                new { message = ex.Message });
        }
        catch (DirectoryNotFoundException ex)
        {
            return NotFound(new { message = ex.Message });
        }
        catch (ArgumentException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
        catch (HttpRequestException)
        {
            return StatusCode(
                StatusCodes.Status503ServiceUnavailable,
                new
                {
                    message =
                        "Python model_runtime 서버에 연결할 수 없습니다. " +
                        "8000번 서버를 확인해주세요."
                });
        }
    }

    [HttpGet("jobs/{jobId}")]
    public async Task<IActionResult> GetJob(
        string jobId,
        CancellationToken cancellationToken)
    {
        var guard = GuardDebugEndpoint();
        if (guard is not null)
            return guard;

        try
        {
            return Ok(await _analyzer.GetJobAsync(jobId, cancellationToken));
        }
        catch (AnalyzerApiException ex)
        {
            return StatusCode((int)ex.StatusCode, new { message = ex.Message });
        }
        catch (HttpRequestException)
        {
            return StatusCode(
                StatusCodes.Status503ServiceUnavailable,
                new { message = "Python model_runtime 서버에 연결할 수 없습니다." });
        }
    }

    [HttpGet("jobs/{jobId}/analysis")]
    public async Task<IActionResult> GetAnalysis(
        string jobId,
        CancellationToken cancellationToken)
    {
        var guard = GuardDebugEndpoint();
        if (guard is not null)
            return guard;

        try
        {
            var json = await _analyzer.GetAnalysisJsonAsync(
                jobId,
                cancellationToken);

            return Content(json, "application/json");
        }
        catch (AnalyzerApiException ex)
        {
            return StatusCode((int)ex.StatusCode, new { message = ex.Message });
        }
        catch (HttpRequestException)
        {
            return StatusCode(
                StatusCodes.Status503ServiceUnavailable,
                new { message = "Python model_runtime 서버에 연결할 수 없습니다." });
        }
    }

    private IActionResult? GuardDebugEndpoint()
    {
        if (!_environment.IsDevelopment() ||
            !_configuration.GetValue<bool>("RuntimeDebug:Enabled"))
        {
            return NotFound(new
            {
                message = "Runtime debug API is disabled."
            });
        }

        var remoteAddress = HttpContext.Connection.RemoteIpAddress;
        if (remoteAddress is null || !IPAddress.IsLoopback(remoteAddress))
        {
            return StatusCode(
                StatusCodes.Status403Forbidden,
                new
                {
                    message =
                        "Runtime debug API can only be called from localhost."
                });
        }

        return null;
    }

    private string ResolveAllowedRoot()
    {
        return LocalProjectFileCollector.ResolveConfiguredRoot(
            _environment.ContentRootPath,
            _configuration["RuntimeDebug:AllowedProjectRoot"]);
    }

    private static long Megabytes(int value) =>
        checked((long)value * 1024L * 1024L);
}

public sealed record RuntimePathAnalysisRequest(
    string ProjectPath,
    string? ProjectName = null,
    double Sensitivity = 0.5,
    string? Model = null,
    string? ApiKey = null);
