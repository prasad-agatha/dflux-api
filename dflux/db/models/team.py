from django.db import models
from dflux.db.mixins import TimeAuditModel
from django.contrib.auth.models import User


class Team(TimeAuditModel):
    """
    This model will allows store all the Team data in Team table.
    """

    name = models.CharField(max_length=256, unique=True)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "Team"
        verbose_name_plural = "Teams"
        db_table = "teams"

    def __str__(self):
        return self.name


class TeamMembers(TimeAuditModel):
    """
    This model will allows store all the TeamMembers data in TeamMembers table.

    * This model contains FK(one to many) relation with User, Team models.
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    team = models.ForeignKey(Team, on_delete=models.CASCADE)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "TeamMembers"
        verbose_name_plural = "TeamMembers"
        db_table = "teammembers"
        unique_together = ("user", "team")

    def __str__(self):
        return f"{self.user.username}"


class TeamInvitation(TimeAuditModel):
    """
    This model will allows store all the TeamInvitation data in TeamInvitation table.

    * This model contains FK(one to many) relation with Team model.
    """

    user = models.CharField(max_length=256)
    team = models.ForeignKey(Team, on_delete=models.CASCADE)
    token = models.CharField(max_length=1000)
    status = models.BooleanField(default=True)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "TeamInvitation"
        verbose_name_plural = "TeamInvitations"
        db_table = "teaminvitations"

    def __str__(self):
        return self.user
