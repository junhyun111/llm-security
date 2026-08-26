using LlmSecurity.Api.DTOs;
using LlmSecurity.Api.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;

namespace LlmSecurity.Api.Controllers;

[ApiController]
[Route("api/auth")]
public class AuthController : ControllerBase
{
    private readonly UserManager<AppUser> _userManager;
    private readonly SignInManager<AppUser> _signInManager;

    public AuthController(
        UserManager<AppUser> userManager,
        SignInManager<AppUser> signInManager)
    {
        _userManager = userManager;
        _signInManager = signInManager;
    }

    [HttpPost("register")]
    public async Task<IActionResult> Register(RegisterRequest request)
    {
        var email = request.Email.Trim().ToLowerInvariant();
        var displayName = request.DisplayName.Trim();

        if (string.IsNullOrWhiteSpace(displayName))
            return BadRequest(new { message = "이름을 입력해주세요." });

        var user = new AppUser
        {
            UserName = email,
            Email = email,
            DisplayName = displayName
        };

        var result = await _userManager.CreateAsync(user, request.Password);

        if (!result.Succeeded)
        {
            return BadRequest(new
            {
                message = "회원가입에 실패했습니다.",
                errors = result.Errors.Select(x => x.Description)
            });
        }

        await _signInManager.SignInAsync(user, isPersistent: false);

        return Ok(new UserResponse(
            user.Id,
            user.Email ?? "",
            user.DisplayName));
    }

    [HttpPost("login")]
    public async Task<IActionResult> Login(LoginRequest request)
    {
        var email = request.Email.Trim().ToLowerInvariant();
        var user = await _userManager.FindByEmailAsync(email);

        if (user is null)
            return Unauthorized(new { message = "이메일 또는 비밀번호가 올바르지 않습니다." });

        var result = await _signInManager.PasswordSignInAsync(
            user,
            request.Password,
            isPersistent: false,
            lockoutOnFailure: true);

        if (!result.Succeeded)
            return Unauthorized(new { message = "이메일 또는 비밀번호가 올바르지 않습니다." });

        return Ok(new UserResponse(
            user.Id,
            user.Email ?? "",
            user.DisplayName));
    }

    [Authorize]
    [HttpPost("logout")]
    public async Task<IActionResult> Logout()
    {
        await _signInManager.SignOutAsync();
        return NoContent();
    }

    [Authorize]
    [HttpGet("me")]
    public async Task<IActionResult> Me()
    {
        var user = await _userManager.GetUserAsync(User);
        if (user is null)
            return Unauthorized();

        return Ok(new UserResponse(
            user.Id,
            user.Email ?? "",
            user.DisplayName));
    }
}
