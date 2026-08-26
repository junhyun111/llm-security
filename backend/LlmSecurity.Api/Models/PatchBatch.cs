namespace LlmSecurity.Api.Models;

public class PatchBatch
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public Guid AnalysisJobId { get; set; }
    public AnalysisJob? AnalysisJob { get; set; }

    public string AnalyzerPatchId { get; set; } = "";
    public string Status { get; set; } = "";
    public string Summary { get; set; } = "";
    public string UnifiedDiff { get; set; } = "";
    public string FindingIdsJson { get; set; } = "[]";

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}
