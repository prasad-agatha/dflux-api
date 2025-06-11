from email.policy import default
from django.db import models
from django.contrib.auth.models import User

from ..mixins import TimeAuditModel


class Profile(TimeAuditModel):
    """
    This model will allows store all the Profile data in Profile table.

    * This model contains FK(one to many) relation with User model.
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=256)
    token_status = models.BooleanField(default=True)
    email_verified = models.BooleanField(default=False)
    company = models.CharField(max_length=256, null=True, blank=True)
    role = models.CharField(max_length=256, null=True, blank=True)
    contact_number = models.CharField(max_length=256, null=True, blank=True)
    industry = models.CharField(max_length=256, null=True, blank=True)
    profile_pic = models.URLField(null=True, blank=True)
    timezone = models.CharField(max_length=256, blank=True, null=True)
    extended_date = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "profile"
        verbose_name_plural = "profiles"
        db_table = "profiles"

    def __str__(self):
        return self.user.username
