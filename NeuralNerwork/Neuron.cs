using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType Type { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }
        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            if (inputCount < 0)
            {
                throw new Exception("Bad input error: inputCount is less than 0");
            }
            Type = type;
            Weights = new List<double>();
            Inputs = new List<double>();
            InitWithRandomValues(inputCount);
        }

        private void InitWithRandomValues(int inputCount)
        {

            var random = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(Type == NeuronType.Input ? 1 : random.NextDouble());
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs, bool show = false)
        {
            if (inputs.Count != Weights.Count)
            {
                throw new Exception("Bad input error: number of weights is not equal to the number of inputs");
            }
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }
            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += Inputs[i] * Weights[i];
            }
            Output = Type == NeuronType.Input ? sum : Sigm(sum);
            return Output;
        }
        public void Learn(double error, double learningRate)
        {
            if(Type == NeuronType.Input)
            {
                return;
            }
            Delta = error * SigmDx(Output);
            for (int i = 0; i < Weights.Count; i++)
            {
                var newWeight = Weights[i] - Inputs[i] * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }
        private double Sigm(double x) => 1.0 / (1.0 + Math.Pow(Math.E, -x));
        private double SigmDx(double x) => Sigm(x) / (1 - Sigm(x));
        public override string ToString() => Output.ToString();
    }
}
