# Generated by Django 3.1.5 on 2021-12-07 16:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0085_project_token'),
    ]

    operations = [
        migrations.AddField(
            model_name='charttrigger',
            name='created_by',
            field=models.CharField(blank=True, max_length=256, null=True),
        ),
    ]
