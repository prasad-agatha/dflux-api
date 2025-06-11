from django.db import models
from django.contrib.auth.models import User

from ..mixins import TimeAuditModel


class UserPasswordResetTokens(TimeAuditModel):
    """
    This model will allows store all the UserPasswordResetTokens data in UserPasswordResetTokens table.

    * This model contains FK(one to many) relation with User model.
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=256)
    token_status = models.BooleanField(default=True)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "UserPasswordResetTokens"
        verbose_name_plural = "UserPasswordResetTokens"
        db_table = "userpasswordresettokens"

    def __str__(self):
        return self.user.username
