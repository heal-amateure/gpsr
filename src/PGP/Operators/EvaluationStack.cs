using GPSR.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace GPSR.Core.Operators {
  public static partial class Evaluation {

    public static double EvaluateStack(Algorithm pgp, RPN<Symbol> program, Task task, DataRecord data) {
      var localEvaluationBuffer = new Stack<double>();
      int targetIdx = task.VariableIndices[task.TargetVariable];

      for (int i = 0; i < data.RowCount; i++) {
        foreach (var symbol in program) {
          if (symbol.Type == SymbolType.Constant) {
            localEvaluationBuffer.Push(symbol.Con.Value);
          }
          else if (symbol.Type == SymbolType.Variable) {
            localEvaluationBuffer.Push(data.Data[symbol.Var.Index * data.RowCount + i] * symbol.Var.Coefficient);
          }
          else {
            var tmpResult = symbol.Opr.Term(localEvaluationBuffer);
            if (double.IsNaN(tmpResult) || double.IsInfinity(tmpResult) || double.IsNegativeInfinity(tmpResult)) {
              localEvaluationBuffer.Clear();
              return double.NaN;
            }
            else {
              localEvaluationBuffer.Push(tmpResult);
            }
            //localEvaluationBuffer.Push(symbol.Opr.Function(localEvaluationBuffer));

            //if (operation.Arity == 1) evaluationBuffer.Push(operation.Function(new[] {evaluationBuffer.Pop()}));
            //else evaluationBuffer.Push(operation.Function(new[] { evaluationBuffer.Pop(), evaluationBuffer.Pop() }));
          }
        }
        var result = localEvaluationBuffer.Pop();
        if (localEvaluationBuffer.Count > 0) {
          Console.WriteLine("\n!!! ERROR !!!\n");
          localEvaluationBuffer.Clear();
          return double.NaN;
        }
        if (double.IsNaN(result) || double.IsInfinity(result) || double.IsNegativeInfinity(result)) {
          return double.NaN;
        }


        program.TrueResults[i] = data.Data[targetIdx * data.RowCount + i]; // not necessary to do this in every evaluation, but it is more convenient to have the true values stored in the program for later use (e.g. for statistics)
        program.EstimatedResults[i] = result;
      }
      program.Score = task.Score.Compute(program);
      return program.Score;
    }

    // Computes the program output for every row without needing a target variable.
    // Unlike EvaluateStack, an invalid row yields NaN instead of aborting the evaluation.
    public static double[] PredictStack(RPN<Symbol> program, DataRecord data) {
      var localEvaluationBuffer = new Stack<double>();
      var results = new double[data.RowCount];

      for (int i = 0; i < data.RowCount; i++) {
        localEvaluationBuffer.Clear();
        foreach (var symbol in program) {
          if (symbol.Type == SymbolType.Constant) {
            localEvaluationBuffer.Push(symbol.Con.Value);
          }
          else if (symbol.Type == SymbolType.Variable) {
            localEvaluationBuffer.Push(data.Data[symbol.Var.Index * data.RowCount + i] * symbol.Var.Coefficient);
          }
          else {
            localEvaluationBuffer.Push(symbol.Opr.Term(localEvaluationBuffer));
          }
        }
        var result = localEvaluationBuffer.Count == 1 ? localEvaluationBuffer.Pop() : double.NaN;
        results[i] = double.IsFinite(result) ? result : double.NaN;
      }
      return results;
    }

  }
}
