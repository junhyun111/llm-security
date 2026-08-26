using System.Net;
using System.Text.Json;
using LlmSecurity.Api.Data;
using LlmSecurity.Api.DTOs;
using LlmSecurity.Api.Models;
using LlmSecurity.Api.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace LlmSecurity.Api.Controllers;

[Authorize]
[ApiController]
[Route("api/analyses")]
public class AnalysesController : ControllerBase
{
    private readonly ApplicationDbContext _db;
    private readonly UserManager<AppUser> _userManager;
    private readonly PythonAnalyzerClient _analyzer;
    private readonly AnalysisSyncService _sync;

    public AnalysesController(
        ApplicationDbContext db,
        UserManager<AppUser> userManager,
        PythonAnalyzerClient analyzer,
        AnalysisSyncService sync)
    {
        _db = db;
        _userManager = userManager;
        _analyzer = analyzer;
        _sync = sync;
    }

    [HttpGet]
    public async Task<ActionResult<List<AnalysisJobResponse>>> List(
        CancellationToken cancellationToken)
    {
        var userId = _userManager.GetUserId(User)!;

        var entities = await _db.AnalysisJobs
            .AsNoTracking()
            .Where(x => x.UserId == userId)
            .OrderByDescending(x => x.CreatedAt)
            .ToListAsync(cancellationToken);

        return Ok(entities.Select(ToResponse).ToList());
    }

    [HttpPost]
    [RequestFormLimits(MultipartBodyLengthLimit = 1_200_000_000)]
    [RequestSizeLimit(1_200_000_000)]
    public async Task<ActionResult<AnalysisJobResponse>> Create(
        [FromForm(Name = "project_name")] string projectName,
        [FromForm(Name = "relative_paths")] List<string> relativePaths,
        [FromForm(Name = "files")] List<IFormFile> files,
        CancellationToken cancellationToken)
    {
        if (files.Count == 0)
            return BadRequest(new { message = "분석할 파일을 선택해주세요." });

        if (files.Count != relativePaths.Count)
            return BadRequest(new { message = "파일과 상대 경로 개수가 일치하지 않습니다." });

        try
        {
            var remote = await _analyzer.CreateJobAsync(
                projectName,
                files,
                relativePaths,
                cancellationToken);

            var job = new AnalysisJob
            {
                UserId = _userManager.GetUserId(User)!,
                AnalyzerJobId = remote.JobId,
                ProjectName = remote.ProjectName,
                Status = remote.Status,
                Progress = remote.Progress,
                Message = remote.Message,
                FileCount = remote.FileCount,
                SourceFileCount = remote.SourceFileCount,
                FindingCount = remote.FindingCount,
                ValidatedFindingCount = remote.ValidatedFindingCount,
                TotalCost = remote.TotalCost,
                ErrorMessage = remote.Error
            };

            _db.AnalysisJobs.Add(job);
            await _db.SaveChangesAsync(cancellationToken);

            return Accepted(ToResponse(job));
        }
        catch (AnalyzerApiException ex)
        {
            return StatusCode((int)ex.StatusCode, new { message = ex.Message });
        }
        catch (HttpRequestException)
        {
            return StatusCode(
                StatusCodes.Status503ServiceUnavailable,
                new { message = "Python 분석 서버에 연결할 수 없습니다. 분석 서버가 실행 중인지 확인해주세요." });
        }
    }

    [HttpGet("{id:guid}")]
    public async Task<ActionResult<AnalysisDetailResponse>> Get(
        Guid id,
        CancellationToken cancellationToken)
    {
        var job = await GetOwnedJob(id, cancellationToken);
        if (job is null)
            return NotFound();

        try
        {
            if (job.Status is "uploading" or "queued" or "analyzing" ||
                job.AnalysisJson is null)
            {
                await _sync.SyncAsync(job, cancellationToken);
            }
        }
        catch (AnalyzerApiException ex) when (ex.StatusCode == HttpStatusCode.Conflict)
        {
            // 분석 진행 중이면 DB에 저장된 상태를 그대로 응답한다.
        }
        catch (HttpRequestException)
        {
            // 분석기 일시 중단 시에도 기존 DB 이력은 조회할 수 있게 한다.
        }

        JsonElement? analysis = null;

        if (!string.IsNullOrWhiteSpace(job.AnalysisJson))
        {
            using var doc = JsonDocument.Parse(job.AnalysisJson);
            analysis = doc.RootElement.Clone();
        }

        return Ok(new AnalysisDetailResponse(ToResponse(job), analysis));
    }

    [HttpPost("{id:guid}/patches/proposal")]
    public async Task<IActionResult> ProposePatch(
        Guid id,
        PatchProposalRequest request,
        CancellationToken cancellationToken)
    {
        var job = await GetOwnedJob(id, cancellationToken);
        if (job is null)
            return NotFound();

        if (request.FindingIds.Count == 0)
            return BadRequest(new { message = "수정할 취약점을 하나 이상 선택해주세요." });

        try
        {
            var patchJson = await _analyzer.ProposePatchAsync(
                job.AnalyzerJobId,
                request.FindingIds,
                cancellationToken);

            await _sync.StorePatchResponseAsync(job, patchJson, cancellationToken);

            return Content(patchJson, "application/json");
        }
        catch (AnalyzerApiException ex)
        {
            return StatusCode((int)ex.StatusCode, new { message = ex.Message });
        }
    }

    [HttpPost("{id:guid}/patches/{patchId}/{action}")]
    public async Task<IActionResult> PatchAction(
        Guid id,
        string patchId,
        string action,
        CancellationToken cancellationToken)
    {
        var job = await GetOwnedJob(id, cancellationToken);
        if (job is null)
            return NotFound();

        if (action is not ("approve" or "reject"))
            return BadRequest(new { message = "지원하지 않는 패치 작업입니다." });

        try
        {
            var patchJson = await _analyzer.PatchActionAsync(
                job.AnalyzerJobId,
                patchId,
                action,
                cancellationToken);

            await _sync.StorePatchResponseAsync(job, patchJson, cancellationToken);

            return Content(patchJson, "application/json");
        }
        catch (AnalyzerApiException ex)
        {
            return StatusCode((int)ex.StatusCode, new { message = ex.Message });
        }
    }

    [HttpGet("{id:guid}/download")]
    public async Task<IActionResult> Download(
        Guid id,
        CancellationToken cancellationToken)
    {
        var job = await GetOwnedJob(id, cancellationToken);
        if (job is null)
            return NotFound();

        try
        {
            using var response = await _analyzer.DownloadAsync(
                job.AnalyzerJobId,
                cancellationToken);

            var bytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);
            var safeName = string.Concat(job.ProjectName.Select(c =>
                char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '-')).Trim('-');

            if (string.IsNullOrWhiteSpace(safeName))
                safeName = "project";

            return File(
                bytes,
                "application/zip",
                $"{safeName}-reviewed.zip");
        }
        catch (AnalyzerApiException ex)
        {
            return StatusCode((int)ex.StatusCode, new { message = ex.Message });
        }
    }

    private async Task<AnalysisJob?> GetOwnedJob(
        Guid id,
        CancellationToken cancellationToken)
    {
        var userId = _userManager.GetUserId(User)!;

        return await _db.AnalysisJobs
            .SingleOrDefaultAsync(
                x => x.Id == id && x.UserId == userId,
                cancellationToken);
    }

    private static AnalysisJobResponse ToResponse(AnalysisJob x) => new(
        x.Id,
        x.ProjectName,
        x.Status,
        x.Progress,
        x.Message,
        x.FileCount,
        x.SourceFileCount,
        x.FindingCount,
        x.ValidatedFindingCount,
        x.TotalCost,
        x.ErrorMessage,
        x.CreatedAt,
        x.UpdatedAt,
        x.CompletedAt);
}
