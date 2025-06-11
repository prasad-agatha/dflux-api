from rest_framework import status
from rest_framework.response import Response

from dflux.db.models import Profile
from dflux.api.views.base import BaseAPIView


class VerifyEmailEndpoint(BaseAPIView):
    """
    API endpoint that allows verify the email.
    - if email is verified it will give success response else failure response.

    * Authentication not required.
    * This endpoint will allows only POST method.
    """

    def post(self, request):

        try:
            token = request.data.get("token")
            profile = Profile.objects.get(user=request.user)
            if profile.token == token:
                profile.email_verified = True
                profile.token_status = True
                profile.save()
                return Response(
                    {"message": "email verified"}, status=status.HTTP_200_OK
                )
            else:
                return Response(
                    {"message": "email verification failed"},
                    status=status.HTTP_401_UNAUTHORIZED,
                )

        except Exception as e:
            return Response(
                {"message": f"{e}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
