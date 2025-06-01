# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of our software seriously. If you discover a security vulnerability in Backlog Builder, we appreciate your help in disclosing it to us in a responsible manner.

### How to Report a Vulnerability

Please report security vulnerabilities by emailing our security team at [security@example.com](mailto:security@example.com).

Include the following information in your report:

- A description of the vulnerability
- Steps to reproduce the issue
- Any proof-of-concept code or exploit
- Impact of the vulnerability
- Your name and affiliation (if any)

### Our Commitment

- We will acknowledge receipt of your report within 3 business days
- We will keep you informed about the progress of the fix
- We will credit you in our security advisories (unless you prefer to remain anonymous)

### Responsible Disclosure Policy

We follow responsible disclosure guidelines:

- Allow us time to investigate and address the vulnerability
- Do not exploit the vulnerability for malicious purposes
- Do not disclose the vulnerability publicly until we've had time to address it

### Security Updates

Security updates will be released as patch versions following semantic versioning. Please update to the latest version to ensure you have all security fixes.

## Secure Development

### Dependencies

We regularly update our dependencies to include the latest security patches. You can see our dependency status in the `package.json` files.

### Security Features

- All API endpoints require authentication
- Passwords are hashed using bcrypt
- JWT tokens are used for session management
- Rate limiting is implemented to prevent abuse
- Input validation is performed on all user inputs
- CORS is properly configured

### Reporting Security Issues in Dependencies

If you find a security vulnerability in one of our dependencies, please report it to the appropriate package maintainer first. If the vulnerability affects our application directly, please follow our reporting process above.

## Security Best Practices for Users

- Keep your GitHub access tokens secure and rotate them regularly
- Use strong, unique passwords for all accounts
- Keep your system and dependencies up to date
- Review and understand the permissions you grant to the application
- Monitor your GitHub account for suspicious activity

## Security Updates and Notifications

Security updates will be announced in:
- GitHub Security Advisories
- The project's releases page
- The CHANGELOG.md file

## Contact

For security-related inquiries, please contact [security@example.com](mailto:security@example.com).

## Legal

This security policy is subject to change at any time. By using this software, you agree to the terms outlined in this policy.
