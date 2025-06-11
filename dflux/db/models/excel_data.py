from django.db import models
from dflux.db.mixins import TimeAuditModel
from django.contrib.auth.models import User
from dflux.db.models import Project, Connection


class ExcelData(TimeAuditModel):
    """
    This model will allows store all the ExcelData data in ExcelData table.

    * This model contains FK(one to many) relation with User, Project, Connection models.
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    tablename = models.CharField(max_length=256)
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, null=True, blank=True
    )
    file_type = models.CharField(max_length=256, null=True, blank=True)
    connection = models.ForeignKey(
        Connection, on_delete=models.CASCADE, null=True, blank=True
    )

    class Meta:
        verbose_name = "ExcelData"
        verbose_name_plural = "ExcelDatas"
        db_table = "exceldatas"
        unique_together = ("user", "tablename", "file_type")

    def __str__(self):
        return self.tablename
