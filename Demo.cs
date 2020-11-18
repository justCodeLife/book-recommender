using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace recommender
{
    public class Demo
    {
        private static MLContext _context = new MLContext();
        private static IDataView _dataView;
        private static DataOperationsCatalog.TrainTestData splitData;
        private static ITransformer model;
        private static EstimatorChain<ValueToKeyMappingTransformer> estimator;

        public static void Execute()
        {
            LoadData();
            PreProcessData();
            CreateModel();
            EvaluateModel();
            PredictValue();
        }

        private static void PredictValue()
        {
            var predictEngine = _context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);
            PrintResult(predictEngine.Predict(new InputModel
            {
                UserId = 12, ISBN = "1879384493"
            }));
            PrintResult(predictEngine.Predict(new InputModel
            {
                UserId = 12, ISBN = "425176428"
            }));
        }

        private static void PrintResult(ResultModel result)
        {
            Console.WriteLine(
                $"UserId: {result.UserId} | Book: {result.ISBN} | Score: {result.Score} | Is Recommended: {result.Score > 7}");
        }

        private static void EvaluateModel()
        {
            var predicitions = model.Transform(splitData.TestSet);
            var metrics = _context.Recommendation().Evaluate(predicitions, labelColumnName: nameof(InputModel.Rating));
            Console.WriteLine("R^2: {0} | LossFunction: {1} | MeanAbsoluteError: {2} | MeanSquaredError: {3}",
                metrics.RSquared, metrics.LossFunction, metrics.MeanAbsoluteError, metrics.MeanSquaredError);
        }

        private static void LoadData()
        {
            _dataView = _context.Data.LoadFromTextFile<InputModel>("book-ratings-train-dataset.csv", ',', true);
        }

        private static void PreProcessData()
        {
            estimator = _context.Transforms.Conversion
                .MapValueToKey("Encoded_UserID", nameof(InputModel.UserId))
                .Append(_context.Transforms.Conversion.MapValueToKey("Encoded_Book",
                    nameof(InputModel.ISBN)));

            var preProcessData = estimator.Fit(_dataView).Transform(_dataView);
            splitData = _context.Data.TrainTestSplit(preProcessData, 0.05);
        }

        private static void CreateModel()
        {
            var options = new MatrixFactorizationTrainer.Options
            {
                LabelColumnName = nameof(InputModel.Rating),
                MatrixColumnIndexColumnName = "Encoded_UserID",
                MatrixRowIndexColumnName = "Encoded_Book",
                NumberOfIterations = 100,
                ApproximationRank = 100
            };

            var trainer = _context.Recommendation().Trainers.MatrixFactorization(options);
            var pipeline = estimator.Append(trainer);
            model = pipeline.Fit(splitData.TrainSet);
        }
    }
}