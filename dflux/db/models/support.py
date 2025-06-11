from statistics import mode
from django.db import models
from django.contrib.auth.models import User
from dflux.db.mixins import TimeAuditModel


class Support(TimeAuditModel):
    """
    This model will allows store all the Support data in Support table.

    * This model contains FK(one to many) relation with User model.
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    subject = models.CharField(max_length=256)
    description = models.TextField()
    attachment = models.URLField(null=True, blank=True)

    class Meta:
        verbose_name = "Support"
        verbose_name_plural = "Supports"
        db_table = "supports"

    def __str__(self):
        return self.user.username
