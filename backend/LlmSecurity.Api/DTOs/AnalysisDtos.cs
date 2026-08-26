using System.Text.Json;

namespace LlmSecurity.Api.DTOs;

public record AnalysisJobResponse(
    Guid Id,
    string ProjectName,
    string Status,
    int Progress,
    string Message,
    int FileCount,
    int SourceFileCount,
    int FindingCount,
    int ValidatedFindingCount,
    double TotalCost,
    string? ErrorMessage,
    DateTime CreatedAt,
    DateTime UpdatedAt,
    DateTime? CompletedAt
);

public record AnalysisDetailResponse(
    AnalysisJobResponse Job,
    JsonElement? Analysis
);

public record PatchProposalRequest(List<string> FindingIds);

public record DashboardResponse(
    int TotalScans,
    int CompletedScans,
    int TotalFindings,
    int ValidatedFindings,
    int ApprovedPatches,
    List<AnalysisJobResponse> RecentJobs
);
