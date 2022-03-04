﻿using System;

namespace NeuralNetworks
{
    internal class NeuronFunctionsBuilder
    {
        public Func<double, double> Linear => (s) => s;
        public Func<double, double> Sigmoid => (s) => 1d / (1 + Math.Exp(-s));
    }
}