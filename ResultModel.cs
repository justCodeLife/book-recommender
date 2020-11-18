using Microsoft.ML.Data;

namespace recommender
{
    public class ResultModel : InputModel
    {
        [ColumnName("Score")] public float Score { get; set; }
    }
}