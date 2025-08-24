# Registration Approval Workflow

This document describes the user registration approval system implemented for OpenAIPerf.

## Overview

The registration system now requires admin approval before users can access the platform. This ensures quality control and prevents unauthorized access.

## Workflow Steps

### 1. User Registration Request
- User visits `/register` and fills out the registration form
- System creates user account with `approved=false` status
- User sees confirmation message about pending approval
- Admin receives email notification about new registration request

### 2. Admin Review
- Admin receives email notification with registration details
- Admin logs into admin panel at `/admin/users`
- Admin can see all users with their approval status
- Pending users show "Pending" status badge
- Admin can click "Approve" button to approve registration

### 3. User Approval
- When admin approves registration, user status is updated to `approved=true`
- User receives email notification about approval
- User can now log in with their credentials

### 4. Login Protection
- Only approved users can log in
- Unapproved users are treated as if they don't exist in the system
- This prevents access to any protected features

## Email Notifications

### Registration Request Email (to Admin)
**Subject:** "New Registration Request: {username}"

**Content:**
- Username, email, and organization of requester
- Direct link to admin panel for review
- Professional HTML formatting

### Registration Approval Email (to User)
**Subject:** "Registration Approved - Welcome to OpenAIPerf!"

**Content:**
- Welcome message with account details
- Direct link to login page
- Professional HTML formatting with styling

## Database Schema Changes

### User Table Updates
- Added `approved` field (BOOLEAN, default: FALSE)
- Added `created_at` field (DATETIME)
- Created index on `approved` field for performance

### Migration
- Existing users are automatically set to `approved=true`
- New registrations default to `approved=false`

## Admin Interface Updates

### Users Management (`/admin/users`)
- Added "Status" column showing Approved/Pending badges
- Added "Approve" button for pending users
- Color-coded status indicators:
  - Green: Approved users
  - Yellow: Pending users

## Security Features

### Authentication Protection
- `get_current_user()` only returns approved users
- Unapproved users cannot access any protected routes
- No session creation for unapproved users

### Admin Controls
- Only admins can approve registrations
- Approval action logs to console
- Proper error handling for edge cases

## Email Configuration

### Environment Variables
```bash
EMAIL_ENABLED=true
SMTP_SERVER=mail.privateemail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@openaiperf.org
SMTP_PASSWORD=your-email-password
FROM_EMAIL=notifications@openaiperf.org
```

### Testing Mode
- Set `EMAIL_ENABLED=false` for testing
- Email notifications are logged but not sent
- Allows testing workflow without actual email delivery

## Implementation Files

### Core Files Modified
- `app/models.py` - Added approval fields to User model
- `app/main.py` - Updated registration, authentication, and admin routes
- `app/email_service.py` - Added registration notification methods
- `templates/register.html` - Added success/error message display
- `templates/admin/users.html` - Added approval interface

### New Routes Added
- `POST /admin/user/{user_id}/approve` - Approve user registration

## Usage Examples

### New User Registration
1. User visits `/register`
2. Fills form and submits
3. Sees: "Registration request submitted successfully! Please wait for admin approval..."
4. Admin receives email notification

### Admin Approval Process
1. Admin receives email: "New Registration Request: username"
2. Admin clicks link to `/admin/users`
3. Sees pending user with yellow "Pending" badge
4. Clicks "Approve" button
5. User status changes to green "Approved" badge
6. User receives approval email

### User Login After Approval
1. User receives email: "Registration Approved - Welcome to OpenAIPerf!"
2. User clicks login link
3. Can now log in with original credentials
4. Has full access to platform features

## Benefits

1. **Quality Control** - Only approved users can access the platform
2. **Security** - Prevents unauthorized access and spam accounts
3. **Notification System** - Admins are immediately aware of new registrations
4. **User Experience** - Clear communication about approval status
5. **Audit Trail** - All approvals are logged and trackable

## Testing

The workflow has been thoroughly tested with:
- Registration request notifications
- Approval notifications
- Database schema updates
- Admin interface functionality
- Authentication protection

All tests pass successfully and the system is ready for production use.
