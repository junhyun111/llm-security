using LlmSecurity.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace LlmSecurity.Api.Controllers;

[Authorize]
[ApiController]
[Route("api/runtime")]
public class RuntimeController : ControllerBase
{
    private readonly PythonAnalyzerClient _analyzer;

    public RuntimeController(PythonAnalyzerClient analyzer)
    {
        _analyzer = analyzer;
    }

    [HttpGet]
    public async Task<IActionResult> Get(CancellationToken cancellationToken)
    {
        try
        {
            var json = await _analyzer.GetRuntimeMetadataJsonAsync(cancellationToken);
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
                new { message = "Python 분석 서버에 연결할 수 없습니다." });
        }
    }
}
