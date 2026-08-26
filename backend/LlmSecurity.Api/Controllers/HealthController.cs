using LlmSecurity.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace LlmSecurity.Api.Controllers;

[ApiController]
[Route("api/health")]
public class HealthController : ControllerBase
{
    private readonly PythonAnalyzerClient _analyzer;

    public HealthController(PythonAnalyzerClient analyzer)
    {
        _analyzer = analyzer;
    }

    [HttpGet]
    public async Task<IActionResult> Get(CancellationToken cancellationToken)
    {
        var analyzerHealthy = await _analyzer.IsHealthyAsync(cancellationToken);

        return Ok(new
        {
            status = "ok",
            analyzer = analyzerHealthy ? "ok" : "unavailable"
        });
    }
}
