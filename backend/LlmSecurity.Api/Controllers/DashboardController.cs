using LlmSecurity.Api.Data;
using LlmSecurity.Api.DTOs;
using LlmSecurity.Api.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace LlmSecurity.Api.Controllers;

[Authorize]
[ApiController]
[Route("api/dashboard")]
public class DashboardController : ControllerBase
{
    private readonly ApplicationDbContext _db;
    private readonly UserManager<AppUser> _userManager;

    public DashboardController(
        ApplicationDbContext db,
        UserManager<AppUser> userManager)
    {
        _db = db;
        _userManager = userManager;
    }

    [HttpGet]
    public async Task<ActionResult<DashboardResponse>> Get(CancellationToken cancellationToken)
    {
        var userId = _userManager.GetUserId(User)!;

        var jobs = _db.AnalysisJobs
            .AsNoTracking()
            .Where(x => x.UserId == userId);

        var totalScans = await jobs.CountAsync(cancellationToken);
        var completedScans = await jobs.CountAsync(
            x => x.Status == "completed",
            cancellationToken);
        var totalFindings = await jobs.SumAsync(
            x => (int?)x.FindingCount,
            cancellationToken) ?? 0;
        var validatedFindings = await jobs.SumAsync(
            x => (int?)x.ValidatedFindingCount,
            cancellationToken) ?? 0;

        var approvedPatches = await _db.PatchBatches
            .AsNoTracking()
            .CountAsync(
                x => x.AnalysisJob!.UserId == userId && x.Status == "approved",
                cancellationToken);

        var recentEntities = await jobs
            .OrderByDescending(x => x.CreatedAt)
            .Take(5)
            .ToListAsync(cancellationToken);

        var recent = recentEntities.Select(ToResponse).ToList();

        return Ok(new DashboardResponse(
            totalScans,
            completedScans,
            totalFindings,
            validatedFindings,
            approvedPatches,
            recent));
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
