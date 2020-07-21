using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    class Program
    {
        static void Main()
        {
            var dataset = new List<Tuple<double, double[]>>
            {
                new Tuple<double, double[]> (0, new double[] {0,0,0,0}),
                new Tuple<double, double[]> (0, new double[] {0,0,0,1}),
                new Tuple<double, double[]> (1, new double[] {0,0,1,0}),
                new Tuple<double, double[]> (0, new double[] {0,0,1,1}),
                new Tuple<double, double[]> (0, new double[] {0,1,0,0}),
                new Tuple<double, double[]> (0, new double[] {0,1,0,1}),
                new Tuple<double, double[]> (1, new double[] {0,1,1,0}),
                new Tuple<double, double[]> (0, new double[] {0,1,1,1}),
                new Tuple<double, double[]> (1, new double[] {1,0,0,0}),
                new Tuple<double, double[]> (1, new double[] {1,0,0,1}),
                new Tuple<double, double[]> (1, new double[] {1,0,1,0}),
                new Tuple<double, double[]> (1, new double[] {1,0,1,1}),
                new Tuple<double, double[]> (1, new double[] {1,1,0,0}),
                new Tuple<double, double[]> (0, new double[] {1,1,0,1}),
                new Tuple<double, double[]> (1, new double[] {1,1,1,0}),
                new Tuple<double, double[]> (1, new double[] {1,1,1,1}),
            };
            var topology = new Topology(4, 1, 0.1 ,2);
            var neuralNetwork = new NeuralNetwork(topology);

            var difference = neuralNetwork.Learn(dataset, 100000);

            var result = new List<double>();
            foreach (var data in dataset)
            {
                var r = neuralNetwork.FeedForward(data.Item2).Output;
                result.Add(r);
            }
            for (int i = 0; i < result.Count; i++)
            {
                var expected = Math.Round(dataset[i].Item1, 4);
                var actual = Math.Round(result[i], 4);
                Console.WriteLine($"case {i}: expected = {expected}, actual = {actual}." + (expected == actual ? "" : " Error!!!"));
            }

            Console.WriteLine(neuralNetwork.FeedForward(1, 1, 1, 0));
            Console.WriteLine("Signals");
        }
    }
}
