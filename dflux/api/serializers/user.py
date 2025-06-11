from rest_framework import serializers
from django.contrib.auth.models import User
from dflux.db.models import UserPasswordResetTokens, Profile

from .base import BaseModelSerializer


class UserSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the User model data.
    """

    class Meta:
        model = User
        fields = (
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "date_joined",
            "is_active",
            "is_superuser",
        )

    def validate_email(self, email):
        if User.objects.filter(email=email).exists():
            raise serializers.ValidationError("Email already exists")
        return email


class PasswordResetTokenSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the UserPasswordResetTokens model data.
    """

    class Meta:
        model = UserPasswordResetTokens
        fields = "__all__"


class ProfileSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Profile model data.
    """

    class Meta:
        model = Profile
        fields = "__all__"


class UsersListSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the User model data.
    """

    contact_number = serializers.SerializerMethodField()
    company = serializers.SerializerMethodField()
    role = serializers.SerializerMethodField()
    end_date = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = (
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "date_joined",
            "is_active",
            "is_superuser",
            "contact_number",
            "company",
            "role",
            "end_date",
        )

    def get_contact_number(self, user):
        profile = Profile.objects.select_related("user").filter(user=user).first()
        if profile is not None:
            return profile.contact_number

    def get_company(self, user):
        profile = Profile.objects.select_related("user").filter(user=user).first()
        if profile is not None:
            return profile.company

    def get_role(self, user):
        profile = Profile.objects.select_related("user").filter(user=user).first()
        if profile is not None:
            return profile.role

    def get_end_date(self, user):
        profile = Profile.objects.select_related("user").filter(user=user).first()
        if profile is not None:
            return profile.extended_date
