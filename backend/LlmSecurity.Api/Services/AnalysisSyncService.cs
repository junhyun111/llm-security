using System.Text.Json;
using LlmSecurity.Api.Data;
using LlmSecurity.Api.Models;
using Microsoft.EntityFrameworkCore;

namespace LlmSecurity.Api.Services;

public class AnalysisSyncService
{
    private readonly ApplicationDbContext _db;
    private readonly PythonAnalyzerClient _analyzer;

    public AnalysisSyncService(
        ApplicationDbContext db,
        PythonAnalyzerClient analyzer)
    {
        _db = db;
        _analyzer = analyzer;
    }

    public async Task<AnalysisJob> SyncAsync(
        AnalysisJob job,
        CancellationToken cancellationToken = default)
    {
        var remote = await _analyzer.GetJobAsync(job.AnalyzerJobId, cancellationToken);

        job.Status = remote.Status;
        job.Progress = remote.Progress;
        job.Message = remote.Message;
        job.FileCount = remote.FileCount;
        job.SourceFileCount = remote.SourceFileCount;
        job.FindingCount = remote.FindingCount;
        job.ValidatedFindingCount = remote.ValidatedFindingCount;
        job.TotalCost = remote.TotalCost;
        job.ErrorMessage = remote.Error;
        job.UpdatedAt = DateTime.UtcNow;

        if (remote.Status == "completed")
        {
            job.CompletedAt ??= DateTime.UtcNow;

            var analysisJson = await _analyzer.GetAnalysisJsonAsync(
                job.AnalyzerJobId,
                cancellationToken);

            job.AnalysisJson = analysisJson;
            await ReplaceFindingsAsync(job, analysisJson, cancellationToken);
            await UpsertPatchBatchFromAnalysisAsync(job, analysisJson, cancellationToken);
        }

        await _db.SaveChangesAsync(cancellationToken);
        return job;
    }

    public async Task StorePatchResponseAsync(
        AnalysisJob job,
        string patchJson,
        CancellationToken cancellationToken = default)
    {
        using var doc = JsonDocument.Parse(patchJson);
        var root = doc.RootElement;

        var patch = await _db.PatchBatches
            .SingleOrDefaultAsync(x => x.AnalysisJobId == job.Id, cancellationToken);

        patch ??= new PatchBatch { AnalysisJobId = job.Id };

        patch.AnalyzerPatchId = GetString(root, "patch_id");
        patch.Status = GetString(root, "status");
        patch.Summary = GetString(root, "summary");
        patch.UnifiedDiff = GetString(root, "unified_diff");
        patch.FindingIdsJson = root.TryGetProperty("finding_ids", out var ids)
            ? ids.GetRawText()
            : "[]";
        patch.UpdatedAt = DateTime.UtcNow;

        if (patch.Id == Guid.Empty || _db.Entry(patch).State == EntityState.Detached)
            _db.PatchBatches.Add(patch);

        await _db.SaveChangesAsync(cancellationToken);

        // Python의 /analysis는 patch_batch도 포함하므로 최신 스냅샷을 다시 저장한다.
        try
        {
            var analysisJson = await _analyzer.GetAnalysisJsonAsync(
                job.AnalyzerJobId,
                cancellationToken);

            job.AnalysisJson = analysisJson;
            await _db.SaveChangesAsync(cancellationToken);
        }
        catch
        {
            // 패치 자체 처리는 성공했으므로 스냅샷 갱신 실패가 응답 전체를 실패시키지 않게 한다.
        }
    }

    private async Task ReplaceFindingsAsync(
        AnalysisJob job,
        string analysisJson,
        CancellationToken cancellationToken)
    {
        var existing = await _db.AnalysisFindings
            .Where(x => x.AnalysisJobId == job.Id)
            .ToListAsync(cancellationToken);

        _db.AnalysisFindings.RemoveRange(existing);

        using var doc = JsonDocument.Parse(analysisJson);
        if (!doc.RootElement.TryGetProperty("findings", out var findings) ||
            findings.ValueKind != JsonValueKind.Array)
            return;

        foreach (var bundle in findings.EnumerateArray())
        {
            if (!bundle.TryGetProperty("finding", out var finding))
                continue;

            bundle.TryGetProperty("validation", out var validation);

            var cwesJson = finding.TryGetProperty("cwes", out var cwes)
                ? cwes.GetRawText()
                : "[]";

            var expertsJson = finding.TryGetProperty("supporting_experts", out var experts)
                ? experts.GetRawText()
                : "[]";

            _db.AnalysisFindings.Add(new AnalysisFinding
            {
                AnalysisJobId = job.Id,
                AnalyzerFindingId = GetString(finding, "finding_id"),
                Title = GetString(finding, "title"),
                FilePath = GetString(finding, "file"),
                LineStart = GetInt(finding, "line_start"),
                LineEnd = GetInt(finding, "line_end"),
                FunctionName = GetString(finding, "function"),
                RootCause = GetString(finding, "root_cause"),
                Consequence = GetString(finding, "consequence"),
                Verdict = GetString(validation, "verdict"),
                Confidence = GetDouble(validation, "confidence"),
                CwesJson = cwesJson,
                ExpertsJson = expertsJson
            });
        }
    }

    private async Task UpsertPatchBatchFromAnalysisAsync(
        AnalysisJob job,
        string analysisJson,
        CancellationToken cancellationToken)
    {
        using var doc = JsonDocument.Parse(analysisJson);

        if (!doc.RootElement.TryGetProperty("patch_batch", out var patchElement) ||
            patchElement.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
            return;

        var patch = await _db.PatchBatches
            .SingleOrDefaultAsync(x => x.AnalysisJobId == job.Id, cancellationToken);

        if (patch is null)
        {
            patch = new PatchBatch { AnalysisJobId = job.Id };
            _db.PatchBatches.Add(patch);
        }

        patch.AnalyzerPatchId = GetString(patchElement, "patch_id");
        patch.Status = GetString(patchElement, "status");
        patch.Summary = GetString(patchElement, "summary");
        patch.UnifiedDiff = GetString(patchElement, "unified_diff");
        patch.FindingIdsJson = patchElement.TryGetProperty("finding_ids", out var ids)
            ? ids.GetRawText()
            : "[]";
        patch.UpdatedAt = DateTime.UtcNow;
    }

    private static string GetString(JsonElement element, string property)
    {
        if (element.ValueKind == JsonValueKind.Undefined ||
            !element.TryGetProperty(property, out var value))
            return "";

        return value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? ""
            : value.ToString();
    }

    private static int GetInt(JsonElement element, string property)
        => element.ValueKind != JsonValueKind.Undefined &&
           element.TryGetProperty(property, out var value) &&
           value.TryGetInt32(out var result)
            ? result
            : 0;

    private static double GetDouble(JsonElement element, string property)
        => element.ValueKind != JsonValueKind.Undefined &&
           element.TryGetProperty(property, out var value) &&
           value.TryGetDouble(out var result)
            ? result
            : 0;
}
