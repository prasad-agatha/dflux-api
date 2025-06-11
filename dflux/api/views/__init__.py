from .project import (
    ProjectView,
    ProjectDetailView,
    ProjectTeamView,
    ProjectTeamDetailView,
    ProjectInvitations,
    ProjectMembersView,
    ProjectMemberDetailView,
    UserProjectRoleView,
)
from .connection import (
    ConnectionView,
    ConnectionDetailView,
    TestConnectionView,
    TestSnowflakeConnectionView,
    TestBigQueryConnectionView,
)
from .auth import (
    UserDetailView,
    UserSignInView,
    UserSignUpView,
    UsersListView,
    ActivateUserView,
)
from .query import connection_establishment, query_invoke
from .query_exection import (
    ExecuteQuery,
    SQLQuery,
    QueryDetail,
    Schematable,
)
from .charts import (
    ChartsView,
    ChartsDetailView,
    LookSharedChart,
    ShareChartView,
    SendChartEmail,
    ChartsLimitedView,
)
from .team import TeamView, TeamMembersView, TeamDetailView, SendTeamInvitationToUsers
from .python_code_execution import PythonCodeRunnerView
from .ml.loading import LoadCSVData, LoadDBData
from .ml.preprocessing import (
    DataTypeCorrection,
    AutoRemoveUnwantedcolumns,
    AutoImputer,
    RemoveCorrelatedColumns,
    LabelEncoding,
    OneHotEncoding,
    OrdinalEncoder,
    Adasyn,
    Smote,
    # CategoricalValueEncoder,
    StandardScale,
    MinMaxScale,
    PreprocessingMethods,
    DetectingOutliersByZscore,
    DetectingOutliersByIOR,
    DetectingOutliersByLocalOutlinerFactor,
    DetectingOutliersByIsolationForest,
    DetectingOutliersByEllipiticEnvelope,
    StandardScaler,
    RobustScaler,
    MinMaxScaling,
    SquareRootTransformation,
    CubeRootTransformation,
    LogTransformation,
    SquareTransformation,
    DropDuplicateColumnsRowsAndUniqueValueColumns,
    FeatureGenerationByPCA,
)
from .ml.modelling import (
    TrainAndTestSplit,
    LogisticRegressionModel,
    SupportVectorClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    XGBClassifier,
    KNNClassifier,
    NaiveBayesClassifier,
    MultinomailNBClassifier,
    AdaBoostClassifier,
    MultiLayerPerceptronClassifier,
    ModellingMethods,
)
from .ml.regression import (
    LinearRegression,
    SupportVectorRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor,
    XGBRegressor,
    RegressionMethods,
    KNeighborsRegressor,
    PolynomialRegression,
    LassoRegressor,
    RidgeRegressor,
    ElasticnetRegression,
    SGDRegression,
    GradientboostingRegression,
    # LGBMRegression,
    # CatBoostRegression,
)
from .ml.prediction import Predict
from .trigger import (
    # TriggerView,
    # TriggerDetailView,
    TriggerOutputViews,
    ChartTriggerView,
    ChartTriggerDetailView,
)
from .dashboard import (
    DashBoardView,
    DashBoardDetailView,
    ShareDashBoardView,
    LookSharedDashBoard,
    UpdateDashboardChartDetails,
    SendDashboardEmail,
    LimitedDashBoardView,
)
from .dump_excel import DumpExcelData, DumpExcelDataDetail, GoogleSheetParserEndpoint
from .password import PasswordResetView, PasswordResetConfirmView, ChangePasswordView
from .data_model import DataModelView, DataModeDetailView, DataModelLimitedView
from .dump_json import DumpJsonDataEndpoint, DumpJsonDataDetail
from .google_sheet import DumpGoogleSheetEndpoint, DumpGoogleSheetDetail
from .asset import MediaAssetView, MediaAssetDetailView
from .verify_email import VerifyEmailEndpoint
from .contact import ContactSaleView, ContactSaleDetailView
from .support import SupportView
from .ml.time_series import (
    TimeSeriesPreprocessingEndpoint,
    TimeSeriesMetaDataEndpoint,
    TimeSeriesPredictionEndpoint,
)
from .base import BaseAPIView
