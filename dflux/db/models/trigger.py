from django.db import models
from dflux.db.mixins import TimeAuditModel

from dflux.db.models import Query, Project, Charts


class ChartTrigger(TimeAuditModel):
    """
    This model will allows store all the ChartTrigger data in ChartTrigger table.

    * This model contains FK(one to many) relation with Project, Charts model.
    """

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    chart = models.ForeignKey(Charts, on_delete=models.CASCADE)
    name = models.CharField(max_length=256, unique=True)
    email = models.JSONField(default=list)
    description = models.TextField(null=True, blank=True)
    cron_expression = models.JSONField(default=dict)
    timezone = models.CharField(max_length=256)
    extra = models.JSONField(null=True, blank=True)
    created_by = models.CharField(max_length=256, null=True, blank=True)

    class Meta:
        verbose_name = "ChartTrigger"
        verbose_name_plural = "ChartTriggers"
        db_table = "charttriggers"
        unique_together = ("project", "name")

    def __str__(self):
        return self.name


class Trigger(TimeAuditModel):
    """
    This model will allows store all the Trigger data in Trigger table.

    * This model contains FK(one to many) relation with Project, ChartTrigger, Query models.
    """

    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    charttrigger = models.ForeignKey(ChartTrigger, on_delete=models.CASCADE)
    query = models.ForeignKey(Query, on_delete=models.CASCADE)
    name = models.CharField(max_length=256, unique=True)
    email = models.JSONField(default=list)
    description = models.TextField(null=True, blank=True)
    cron_expression = models.JSONField(default=dict)
    timezone = models.CharField(max_length=256)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "Trigger"
        verbose_name_plural = "Triggers"
        db_table = "triggers"
        unique_together = ("project", "name")

    def __str__(self):
        return self.name


class TriggerOutput(TimeAuditModel):
    """
    This model will allows store all the TriggerOutput data in TriggerOutput table.

    * This model contains FK(one to many) relation with Trigger model.
    """

    trigger = models.ForeignKey(Trigger, on_delete=models.CASCADE)
    data = models.JSONField(default=dict)
    extra = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name = "TriggerOutput"
        verbose_name_plural = "TriggerOutputs"
        db_table = "triggeroutputs"

    def __str__(self):
        return self.trigger.name
