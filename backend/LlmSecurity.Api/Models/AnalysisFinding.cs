namespace LlmSecurity.Api.Models;

public class AnalysisFinding
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public Guid AnalysisJobId { get; set; }
    public AnalysisJob? AnalysisJob { get; set; }

    public string AnalyzerFindingId { get; set; } = "";
    public string Title { get; set; } = "";
    public string FilePath { get; set; } = "";
    public int LineStart { get; set; }
    public int LineEnd { get; set; }
    public string FunctionName { get; set; } = "";
    public string RootCause { get; set; } = "";
    public string Consequence { get; set; } = "";

    public string Verdict { get; set; } = "";
    public double Confidence { get; set; }

    public string CwesJson { get; set; } = "[]";
    public string ExpertsJson { get; set; } = "[]";
}
