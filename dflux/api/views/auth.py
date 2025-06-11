from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser

from django.utils import timezone
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.shortcuts import get_object_or_404


from dflux.db.models import Profile
from dflux.utils.emails import emails
from dflux.api.views.base import BaseAPIView
from dflux.api.serializers import UserSerializer, ProfileSerializer, UsersListSerializer

from .filters import UserFilter
from .utils import generate_signup_token, get_tokens_for_user
from .permissions import DomainValidationPermissions


class UserSignUpView(BaseAPIView):
    """
    API endpoint that allows register new user.

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    def post(self, request):
        import uuid

        try:
            email = request.data.get("email")
            first_name = request.data.get("first_name")
            last_name = request.data.get("last_name")
        except KeyError:
            return Response({"error": "missing required fields"})

        # creating random username
        user = User.objects.create(
            email=email,
            username=uuid.uuid4().hex,
            first_name=first_name,
            last_name=last_name,
        )
        user.set_password(request.data.get("password"))
        user.save()
        token = generate_signup_token()
        # save user registration token
        Profile.objects.create(user=user, token=token)
        emails.send_registration_email(request, token)
        jwt_token = get_tokens_for_user(user)
        jwt_token["email"] = request.data.get("email")
        return Response(jwt_token)


class UserSignInView(BaseAPIView):
    """
    API endpoint that allows to login user.
    - Once user is logged in this will generate new access , refresh tokens.

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    def post(self, request):
        try:
            user = User.objects.get(email=request.data.get("email"))
            if not user.is_active:
                return Response(
                    {"error": "user is inactive please contact your admin"},
                    status=status.HTTP_401_UNAUTHORIZED,
                )
            user_ = authenticate(
                username=user.username, password=request.data.get("password")
            )
            if user_:
                jwt_token = get_tokens_for_user(user)
                return Response(jwt_token)
            else:
                return Response(
                    {"error": "Please check your email and password."},
                    status=status.HTTP_401_UNAUTHORIZED,
                )
        except:
            return Response(
                {"error": "Please check your email and password."},
                status=status.HTTP_401_UNAUTHORIZED,
            )


class UserDetailView(BaseAPIView):
    """
    API endpoint that allows view, update, delete individual user details.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, PUT, DELETE methods.
    """

    permission_classes = (IsAuthenticated,)

    def get(self, request):
        """
        This method allows view individual user details.
        """
        user = request.user
        user_serializer = UserSerializer(user)
        profile = Profile.objects.filter(user=user).first()
        profile_serializer = ProfileSerializer(profile)
        data = user_serializer.data
        data.update(profile_serializer.data)
        return Response(data)

    def put(self, request):
        """
        This method allows update individual user details.
        """
        user = request.user
        user_serializer = UserSerializer(user, data=request.data, partial=True)
        profile = Profile.objects.filter(user=user).first()
        profile_serializer = ProfileSerializer(profile, data=request.data, partial=True)
        if user_serializer.is_valid() and profile_serializer.is_valid():
            user_serializer.save()
            profile_serializer.save()
            data = user_serializer.data
            data.update(profile_serializer.data)
            return Response(data)
        errors = user_serializer.errors
        errors.update(profile_serializer.errors)
        return Response(errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        """
        This method allows delete individual user details.
        """
        user_delete = request.user
        user_delete.delete()
        return Response({"message": "Delete Success"}, status=status.HTTP_200_OK)


class ActivateUserView(BaseAPIView):
    """
    API endpoint that allows user is active or not.
    - If user is active it will return True else it will return False

    * This endpoint will allows only GET method.
    """

    # permission_classes = (IsAdminUser,)

    def get(self, request, id):

        user = get_object_or_404(User, id=id)
        profile = Profile.objects.filter(user=user).first()
        user.is_active = True
        if profile is not None:
            profile.extended_date = timezone.now()
            user.save()
            profile.save()
        return Response({"message": "user activated"})


class UsersListView(BaseAPIView):
    """
    API endpoint that allows view list of all users in the databases.

    * Requires JWT authentication.
    * This endpoint will allows only GET method.
    """

    permission_classes = (IsAuthenticated, DomainValidationPermissions)

    def get(self, request):
        users = UserFilter(request.GET, queryset=User.objects.all()).qs
        serializer = UsersListSerializer(users, many=True)
        return Response(serializer.data)
