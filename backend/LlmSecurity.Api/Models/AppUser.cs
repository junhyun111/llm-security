using Microsoft.AspNetCore.Identity;

namespace LlmSecurity.Api.Models;

public class AppUser : IdentityUser
{
    public string DisplayName { get; set; } = "";
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public ICollection<AnalysisJob> AnalysisJobs { get; set; } = new List<AnalysisJob>();
}
