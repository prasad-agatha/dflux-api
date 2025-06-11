from django.db import models
from dflux.db.mixins import TimeAuditModel
from django.contrib.auth.models import User
from dflux.db.models import Project, Connection


class JsonData(TimeAuditModel):
    """
    This model will allows store all the JsonData data in JsonData table.

    * This model contains FK(one to many) relation with User, Project, Connection models.
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    tablename = models.CharField(max_length=256)
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, null=True, blank=True
    )
    connection = models.ForeignKey(
        Connection, on_delete=models.CASCADE, null=True, blank=True
    )

    class Meta:
        verbose_name = "JsonData"
        verbose_name_plural = "JsonData"
        db_table = "JsonData"
        unique_together = ("user", "tablename")

    def __str__(self):
        return self.tablename
