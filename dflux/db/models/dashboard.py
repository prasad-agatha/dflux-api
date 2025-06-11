from django.db import models
from dflux.db.mixins import TimeAuditModel

from dflux.db.models import Project, Charts
from django.contrib.auth.models import User


class DashBoard(TimeAuditModel):
    """
    This model will allows store all the dashboard data in dashboard table.

    * This model contains FK(one to many) relation with User, Project models.
    """

    name = models.CharField(max_length=256)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    description = models.TextField(null=True, blank=True)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "Dashboard"
        verbose_name_plural = "Dashboards"
        db_table = "dashboards"
        unique_together = ("name", "project")

    def __str__(self):
        return self.name

    @property
    def charts(self):
        return DashBoardCharts.objects.select_related("dashboard", "chart").filter(
            dashboard=self
        )


class DashBoardCharts(TimeAuditModel):
    """
    This model will allows store all the dashboard charts data in dashboard charts table.

    * This model contains FK(one to many) relation with DashBoard, Charts models.
    """

    dashboard = models.ForeignKey(DashBoard, on_delete=models.CASCADE)
    chart = models.ForeignKey(Charts, on_delete=models.CASCADE)
    height = models.CharField(max_length=256, null=True, blank=True)
    width = models.CharField(max_length=256, null=True, blank=True)
    position_x = models.CharField(max_length=256, null=True, blank=True)
    position_y = models.CharField(max_length=256, null=True, blank=True)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "DashBoardCharts"
        verbose_name_plural = "DashBoardCharts"
        db_table = "dashboardCharts"

    def __str__(self):
        return self.dashboard.name


class ShareDashBoard(TimeAuditModel):
    """
    This model will allows store all the ShareDashBoard data in ShareDashBoard charts table.

    * This model contains FK(one to many) relation with DashBoard model.
    """

    dashboard = models.ForeignKey(DashBoard, on_delete=models.CASCADE)
    token = models.CharField(max_length=256)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "ShareDashBoard"
        verbose_name_plural = "ShareDashBoards"
        db_table = "sharedashBoard"

    def __str__(self):
        return self.dashboard.name
