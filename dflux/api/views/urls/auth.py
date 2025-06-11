from django.urls import path

from rest_framework_simplejwt.views import TokenRefreshView

from dflux.api.views import (
    UserDetailView,
    UserSignInView,
    UserSignUpView,
    UsersListView,
    ActivateUserView,
)
from dflux.api.views.verify_email import VerifyEmailEndpoint

from dflux.api.views.password import (
    PasswordResetView,
    PasswordResetConfirmView,
    ChangePasswordView,
)

urlpatterns = [
    path("users/", UserSignUpView.as_view(), name="create"),
    # user login
    path("signin/", UserSignInView.as_view(), name="token_obtain_pair"),
    # user access token refresh
    path("token-refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # get list of users
    path("userslist/", UsersListView.as_view(), name="userslist"),
    # get  particular user details
    path("users/me/", UserDetailView.as_view(), name="retreive user"),
    path(
        "users/<int:id>/activate/",
        ActivateUserView.as_view(),
        name="active-user",
    ),
    # verify email
    path(
        "verify/email/",
        VerifyEmailEndpoint.as_view(),
        name="verify-email",
    ),
    # password
    path("password-reset/", PasswordResetView.as_view(), name="password-reset"),
    path(
        "password-reset/confirm/",
        PasswordResetConfirmView.as_view(),
        name="password-reset-confirm",
    ),
    path(
        "change-password/<int:pk>/",
        ChangePasswordView.as_view(),
        name="auth_change_password",
    ),
]
