using LlmSecurity.Api.Data;
using LlmSecurity.Api.Models;
using LlmSecurity.Api.Services;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

var connectionString = builder.Configuration.GetConnectionString("DefaultConnection")
    ?? "Data Source=data/llm-security.db";

var dbDirectory = Path.GetDirectoryName(
    connectionString.Replace("Data Source=", "", StringComparison.OrdinalIgnoreCase));
if (!string.IsNullOrWhiteSpace(dbDirectory))
    Directory.CreateDirectory(dbDirectory);

builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlite(connectionString));

builder.Services
    .AddIdentity<AppUser, IdentityRole>(options =>
    {
        options.User.RequireUniqueEmail = true;
        options.Password.RequiredLength = 8;
        options.Password.RequireDigit = true;
        options.Password.RequireLowercase = true;
        options.Password.RequireUppercase = false;
        options.Password.RequireNonAlphanumeric = false;

        options.Lockout.MaxFailedAccessAttempts = 5;
        options.Lockout.DefaultLockoutTimeSpan = TimeSpan.FromMinutes(5);
    })
    .AddEntityFrameworkStores<ApplicationDbContext>()
    .AddDefaultTokenProviders();

builder.Services.ConfigureApplicationCookie(options =>
{
    options.Cookie.Name = "llm-security.auth";
    options.Cookie.HttpOnly = true;
    options.Cookie.SameSite = SameSiteMode.Lax;
    options.Cookie.SecurePolicy = CookieSecurePolicy.SameAsRequest;

    // API에서 HTML 로그인 페이지로 리다이렉트하지 않고 401/403을 반환한다.
    options.Events.OnRedirectToLogin = context =>
    {
        context.Response.StatusCode = StatusCodes.Status401Unauthorized;
        return Task.CompletedTask;
    };
    options.Events.OnRedirectToAccessDenied = context =>
    {
        context.Response.StatusCode = StatusCodes.Status403Forbidden;
        return Task.CompletedTask;
    };
});

// PythonAnalyzerClient가 현재 분석 요청의 Form 설정값을 안전하게 읽을 수 있게 한다.
builder.Services.AddHttpContextAccessor();

// 개발용 "폴더 경로 분석"에서 안전하게 로컬 프로젝트 파일을 수집한다.
builder.Services.AddSingleton<LocalProjectFileCollector>();

var frontendOrigin = builder.Configuration["Frontend:Origin"] ?? "http://localhost:5173";
builder.Services.AddCors(options =>
{
    options.AddPolicy("frontend", policy =>
    {
        policy
            .WithOrigins(frontendOrigin)
            .AllowAnyHeader()
            .AllowAnyMethod()
            .AllowCredentials();
    });
});

var analyzerBaseUrl = builder.Configuration["Analyzer:BaseUrl"] ?? "http://127.0.0.1:8000";
builder.Services.AddHttpClient<PythonAnalyzerClient>(client =>
{
    client.BaseAddress = new Uri(analyzerBaseUrl);
    // OpenRouter 분석은 길어질 수 있으므로 넉넉하게 둔다.
    client.Timeout = TimeSpan.FromMinutes(30);
});

builder.Services.AddScoped<AnalysisSyncService>();
builder.Services.AddControllers();

var app = builder.Build();

// 초보 단계에서는 migration 명령 없이 실행 즉시 SQLite DB가 생성되도록 한다.
// 스키마가 안정되면 EnsureCreated 대신 EF Core migrations로 전환하면 된다.
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
    await db.Database.EnsureCreatedAsync();
}

app.UseCors("frontend");

app.UseAuthentication();
app.UseAuthorization();

app.MapControllers();

app.Run();
