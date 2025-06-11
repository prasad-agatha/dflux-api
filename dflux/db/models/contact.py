from django.db import models
from dflux.db.mixins import TimeAuditModel

from django.contrib.auth.models import User


class ContactSale(TimeAuditModel):
    """
    This model will allows store all the ContactSale data in ContactSale table.

    * This model contains FK(one to many) relation with User model.
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    subject = models.CharField(max_length=256)
    message = models.TextField()

    class Meta:
        verbose_name = "ContactSale"
        verbose_name_plural = "ContactSales"
        db_table = "contactsales"

    def __str__(self):
        return self.user.username
