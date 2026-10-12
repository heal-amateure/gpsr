using GPSR.Data;
using GPSR.Utils;
using GPSR.Core;
using System.Diagnostics;
using GPSR.Core.Operators;

namespace GPSR.Runner {
  public class Program {
    public static void Main(string[] args) {
      const int dataSeed = -1;
      const int algorithmSeed = -1;
      var dataRng = new FastRandom(dataSeed);
      var algorithmRng = new FastRandom(algorithmSeed);

      // --- setup sample data set "resinet"
      //var targetVariable = Resinet_TargetVariable_PvProduction;
      //var inputVariables = InputVariables["Resinet_BasicVariableSet_PvProduction"];

      // --- setup sample data set "geotorus"
      var targetVariable = GeoTorus_TargetVariable_Volume;
      var inputVariables = InputVariables["GeoTorus_Volume"];
    
      var allVariables = inputVariables.Append(targetVariable).ToList();
      var variableIndices = allVariables
        .Select((x, i) => new { Item = x, Index = i })
        .ToDictionary(x => x.Item, x => x.Index);

      //DataSet ds = ProtoDataReader.ReadDataset_Numeric(rng, Datasets["Resinet"], allVariables);
      DataSet ds = ProtoDataReader.ReadDataset_Numeric(dataRng, Datasets["GeoTorusLarge"], allVariables);      

      var dds = ds.GetDoubleSet();
      var variableLimitDict = new Dictionary<string, Tuple<double, double>>();

      foreach (var s in dds) {
        var min = s.Value.Values.Min();
        var max = s.Value.Values.Max();
        variableLimitDict.Add(s.Value.Name, Tuple.Create(min, max)); 
      }


      // --- configure data set and modeling task
      DataSet sds = ds.Shuffle(dataRng);
      DataSet trainingSet = sds.Subset(0, 1000);
      DataSet testSet = sds.Subset(2000, 1000);
      Core.Task modelingTask = new Core.Task(
        name: "GPSR",
        inputVariables: inputVariables,
        optimizationTargets: new HashSet<string> { targetVariable, OptimizationTarget.Complexity },
        metric: EvaluationMetric.NMSE,
        optimizationDirection: OptimizationDirection.Minimize        
      );
      modelingTask.VariableLimitsDict = sds.GetDoubleSetLimits();


      // --- configure gp hyperparameters
      var alg = new Algorithm(randomNumberGenerator: algorithmRng,
        generations: 1000,
        populationSize: 100,
        symbolCount: 25,
        nestingDepth: 8,
        crossoverRate: 0.9,
        mutationRate: 0.25,        
        elites: 1);

      // --- configure gp symbol set (grammar)
      alg.SelectedNonterminals = [
        Functions.Addition,
        Functions.Subtraction,
        Functions.Multiplication,
        Functions.AnalyticQuotient,
        Functions.ProtectedLogarithm,
        Functions.ProtectedExponential,
        Functions.Sine,
        Functions.Cosine,
        Functions.Tangent,
        Functions.HyperbolicTangent,
        Functions.Pi
      ];

      alg.SelectedTerminals = [
        Terminal.Variable,
        Terminal.Constant
      ];


      // --- configure gp operators
      alg.Breed = Creation.BreedConstrained;
      alg.Select = Selection.TournamentSelection;
      alg.Optimizer = Optimization.OptimizeCoefficientsAndConstants;
      alg.Crossover = Crossing.Cross;
      alg.Mutators = [Mutation.MutateReplaceSubtree, Mutation.MutateTerminateSubtree];
      alg.Evaluate = Evaluation.EvaluateProgram;


      // --- configure algorithm options
      alg.LogStatistics = true;
      alg.UseParallelization = true;
      alg.UseDeterministicParallelization = true;
      alg.DeterministicSeed = algorithmSeed;
      alg.PerformSimplification = false;
      alg.OptimizationIterations = 10;
      alg.MaxDegreeOfParallelism = -1;


      // --- run gp algorithm
      Console.WriteLine("Starting GPSR...");
      Console.WriteLine("(press any key to stop computation)\n");

      Stopwatch sw = new Stopwatch();
      bool k = false;
      var cts = new CancellationTokenSource();
      sw.Start();
      System.Threading.Tasks.Task t = alg.Fit(modelingTask, trainingSet, cts.Token); // gp algorithm execution

      while (!k && !t.IsCompleted) {
        k = Console.KeyAvailable;
        t.Wait(100);
      }
      cts.Cancel();
      t.Wait(1000);
      sw.Stop();


      // --- print training results/stats
      alg.ComputeScores();

      Console.WriteLine();
      Console.WriteLine();
      Console.WriteLine("Training Results:");
      Console.WriteLine();
      Console.WriteLine($"Evaluations:        {alg.EvaluationCount}");
      Console.WriteLine($"Runtime:            {(sw.ElapsedMilliseconds / 1000.0):f8} seconds");
      Console.WriteLine($"Time / Evaluation:  {(sw.ElapsedMilliseconds / 1000.0 / alg.EvaluationCount):f8} seconds");
      Console.WriteLine();
      Console.WriteLine($"Best Program RPN:   {alg.BestProgramRPN}");
      Console.WriteLine($"Best Program INF:   {alg.BestProgram}");
      Console.WriteLine();
      Console.WriteLine($"Best NMSE:          {alg.BestProgramNMSE}");
      Console.WriteLine($"Best RMSE:          {alg.BestProgramRMSE}");
      Console.WriteLine($"Best MAE:           {alg.BestProgramMAE}");
      Console.WriteLine($"Best MRE:           {alg.BestProgramMRE}");
      Console.WriteLine($"Best Pearson R:     {alg.BestProgramPearsonR}");
      Console.WriteLine($"Best Pearson R2:    {alg.BestProgramPearsonR2}");
      Console.WriteLine($"Best LD:            {alg.BestProgramLD}");
      Console.WriteLine();

      // --- print test results/stats
      alg.ComputeScores(testSet);
      Console.WriteLine("Test Results:");
      Console.WriteLine();
      Console.WriteLine($"Best NMSE:          {alg.BestProgramNMSE}");
      Console.WriteLine($"Best RMSE:          {alg.BestProgramRMSE}");
      Console.WriteLine($"Best MAE:           {alg.BestProgramMAE}");
      Console.WriteLine($"Best MRE:           {alg.BestProgramMRE}");
      Console.WriteLine($"Best Pearson R:     {alg.BestProgramPearsonR}");
      Console.WriteLine($"Best Pearson R2:    {alg.BestProgramPearsonR2}");
      Console.WriteLine($"Best LD:            {alg.BestProgramLD}");
      Console.WriteLine();
    }



    // =============================================================================================    

    public static Dictionary<string, string> Datasets = new Dictionary<string, string>()
    {
       { "Resinet", @"..\..\..\sample-data\resinet.csv" }
      ,{ "GeoTorus", @"..\..\..\sample-data\geo-torus.csv" }
      ,{ "GeoTorusLarge", @"..\..\..\sample-data\geo-torus-large.csv" }
      ,{ "SinglePoint", @"..\..\..\sample-data\single-point.csv" }
      ,{ "SomePoints", @"..\..\..\sample-data\some-points.csv" }      
    };

    public static Dictionary<string, List<string>> InputVariables = new Dictionary<string, List<string>>()
    {
       { "Resinet_BasicVariableSet_PvProduction", new List<string>() { "globalRadiation", "globalRadiationSum1h", "relativeHumidity", "airTemperature", "oneHourPrecipitationSum", "age", "dayLength", "hoursAfterSunrise" } }
      ,{ "Resinet_BasicVariableSet_PowerConsumption", new List<string>() { "globalRadiation", "relativeHumidity", "airTemperature", "oneHourPrecipitationSum", "age", "dayLength", "hoursAfterSunrise" } }
      ,{ "Resinet_BasicVariableSet_BatterySOC", new List<string>() { "globalRadiation", "globalRadiationSum1h", "globalRadiationSum2h", "globalRadiationSum3h", "globalRadiationSumFrame07to12h", "globalRadiationSumFrame13to24h", "relativeHumidity", "airTemperature", "oneHourPrecipitationSum", "age", "dayLength", "hoursAfterSunrise" } }
      ,{ "Resinet_ReducedVariableSet_PvProduction", new List<string>() { "globalRadiation", "age", "hoursAfterSunrise" } }
      ,{ "Resinet_ReducedVariableSet_PowerConsumption", new List<string>() { "dayLength", "hoursAfterSunrise" } }
      ,{ "Resinet_ReducedVariableSet_BatterySOC", new List<string>() { "globalRadiation", "globalRadiationSum2h", "globalRadiationSumFrame07to12h" } }
      ,{ "GeoTorus_Surface", new List<string>() { "r", "R" } }
      ,{ "GeoTorus_Volume", new List<string>() { "r", "R" } }      
    };

    public static string Resinet_TargetVariable_PvProduction = "pvProduction";
    public static string Resinet_TargetVariable_PowerConsumption = "powerConsumption";
    public static string Resinet_TargetVariable_BatterySOC = "batterySOC";
    public static string GeoTorus_TargetVariable_Surface = "A";
    public static string GeoTorus_TargetVariable_Volume = "V";    
  }
}


