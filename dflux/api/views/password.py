import uuid

from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from dflux.api.views.base import BaseAPIView
from dflux.db.models import UserPasswordResetTokens
from dflux.utils.emails import emails
from dflux.api.serializers import PasswordResetTokenSerializer


class PasswordResetView(BaseAPIView):
    """
    API endpoint that allows users reset their passwords when users forgot password.
    - This will send password reset link to email

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    def post(self, request):
        email = request.data.get("email", None)
        user = get_object_or_404(User, email=email)
        uid = str(uuid.uuid4().hex) + str(uuid.uuid4().hex)
        # creating password reset token object
        data = {"user": user.id, "token": uid, "token_status": True}
        serializer = PasswordResetTokenSerializer(data=data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            emails.send_password_reset_email(user, email, uid)
            return Response({"msg": "password reset link set your email address"})


class PasswordResetConfirmView(BaseAPIView):
    """
    API endpoint that allows updated the new password when you get the password reset link.
    - This will update user new password into the database
    - This will send password reset confirmation email

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    def post(self, request):
        try:
            token = request.data.get("token")
            new_password = request.data.get("new_password")
            confirm_password = request.data.get("confirm_password")
            # print(new_password, confirm_password)
        except KeyError:
            return Response(
                {"error": "missing required fields"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = User.objects.get(
                userpasswordresettokens__token=token,
                userpasswordresettokens__token_status=True,
            )
            if new_password != confirm_password:
                return Response(
                    {"error": "new_password, confirm_password must be same"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if not new_password.strip() and not confirm_password.strip():
                return Response(
                    {"error": "new_password, confirm_password should not be empty"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # updating new password
            user.set_password(new_password)
            user.save()

            # set False password token status
            user_passsword_token = UserPasswordResetTokens.objects.get(
                user=user, token=token
            )
            user_passsword_token.token_status = False
            user_passsword_token.save()

            return Response({"message": "your password has been updated successfully"})

        except:
            return Response(
                {"error": "Invalid token"}, status=status.HTTP_400_BAD_REQUEST
            )


class ChangePasswordView(BaseAPIView):
    """
    API endpoint that allows change users existing passwords.
    - This will update user new password into the database

    * Authentication not required.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request, pk):
        try:
            old_password = request.data.get("old_password")
            new_password = request.data.get("new_password")
            confirm_password = request.data.get("confirm_password")
        except KeyError:
            return Response(
                {"error": "missing required fields"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = User.objects.get(id=pk)
        except User.DoesNotExist:
            return Response(
                {"error": "user does not exist"}, status=status.HTTP_400_BAD_REQUEST
            )

        if not user.check_password(old_password):
            return Response(
                {"msg": "old password wrong"}, status=status.HTTP_400_BAD_REQUEST
            )
        if new_password != confirm_password:
            return Response(
                {"error": "new_password, confirm_password must be same"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not new_password.strip() and not confirm_password.strip():
            return Response(
                {"error": "new_password, confirm_password should not be empty"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # updating new password
        user.set_password(new_password)
        user.save()
        return Response({"message": "your password has been updated successfully"})
