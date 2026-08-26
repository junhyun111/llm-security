namespace LlmSecurity.Api.Models;

public class AnalysisJob
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public string UserId { get; set; } = "";
    public AppUser? User { get; set; }

    public string AnalyzerJobId { get; set; } = "";
    public string ProjectName { get; set; } = "";
    public string Status { get; set; } = "queued";
    public int Progress { get; set; }
    public string Message { get; set; } = "";

    public int FileCount { get; set; }
    public int SourceFileCount { get; set; }
    public int FindingCount { get; set; }
    public int ValidatedFindingCount { get; set; }
    public double TotalCost { get; set; }

    public string? ErrorMessage { get; set; }

    // Python 분석 결과의 원본 JSON 스냅샷.
    // Python 결과 형식이 조금 바뀌더라도 기존 분석 기록을 그대로 보존할 수 있다.
    public string? AnalysisJson { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? CompletedAt { get; set; }

    public ICollection<AnalysisFinding> Findings { get; set; } = new List<AnalysisFinding>();
    public PatchBatch? PatchBatch { get; set; }
}
