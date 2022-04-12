using System;
using System.Linq;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.Trainers
{
    public static class LayeredNNTrainingData
    {
        public struct Data
        {
            public double[][] inputs;
            public double[][] results;

            public int inputLayerNeuronCount => inputs[0].Length;
            public int outputLayerNeuronCount => results[0].Length;
        }

        public static Data Plus_Minus = new Data
        {
            inputs = new[]
            {
                new[]
                {
                    //plus
                    0d, 0, 1, 0, 0,
                    0d, 0, 1, 0, 0,
                    1d, 1, 1, 1, 1,
                    0d, 0, 1, 0, 0,
                    0d, 0, 1, 0, 0
                },
                new[]
                {
                    //plus
                    0d, 1, 0, 0, 0,
                    1, 1, 1, 0, 0,
                    0, 1, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                },
                new[]
                {
                    //plus
                    0d, 0, 0, 0, 0,
                    0d, 1, 1, 0, 0,
                    1d, 1, 1, 1, 0,
                    1d, 1, 1, 1, 0,
                    0d, 1, 1, 0, 0
                },
                new[]
                {
                    //minus
                    0d, 0, 1, 1, 1,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                },
                new[]
                {
                    //minus
                    1d, 1, 1, 1, 1,
                    1d, 1, 1, 1, 1,
                    0d, 0, 0, 0, 0,
                    0d, 0, 0, 0, 0,
                    0d, 0, 0, 0, 0
                },
                new[]
                {
                    //minus
                    0d, 0, 0, 0, 0,
                    0d, 0, 0, 0, 0,
                    1d, 1, 0, 0, 0,
                    0d, 0, 1, 1, 1,
                    0d, 0, 0, 0, 0
                },
                new[]
                {
                    //plus
                    0d, 0, 0, 1, 0,
                    0d, 0, 0, 1, 0,
                    0d, 1, 1, 1, 1,
                    0d, 0, 0, 1, 0,
                    0d, 0, 0, 1, 0
                },
                new[]
                {
                    //minus
                    0d, 0, 0, 0, 0,
                    0d, 0, 0, 0, 0,
                    1d, 1, 1, 1, 1,
                    0d, 0, 0, 0, 0,
                    0d, 0, 0, 0, 0,
                }
            },
            results = new[]
            {
                new[] {1d, 0},
                new[] {1d, 0},
                new[] {1d, 0},
                new[] {0d, 1},
                new[] {0d, 1},
                new[] {0d, 1},
                new[] {1d, 0},
                new[] {0d, 1}
            }
        };

        public static Data SQRT_X = new Data
        {
            inputs = Enumerable.Range(0, 20).Select(x => new[] {(double) x}).ToArray(),
            results = Enumerable.Range(0, 20).Select(x => new[] {Math.Sqrt(x)}).ToArray()
        };

        public static Data AND_AB = new Data
        {
            inputs = new[]
            {
                new[] {0d, 0},
                new[] {0d, 1},
                new[] {1d, 0},
                new[] {1d, 1}
            },
            results = new[]
            {
                new[] {0d},
                new[] {0d},
                new[] {0d},
                new[] {1d}
            }
        };

        public static Data XOR_ABC = new Data
        {
            inputs = new[]
            {
                new[] {0d, 0, 0},
                new[] {0d, 0, 1},
                new[] {0d, 1, 0},
                new[] {0d, 1, 1},
                new[] {1d, 0, 0},
                new[] {1d, 0, 1},
                new[] {1d, 1, 0},
                new[] {1d, 1, 1}
            },
            results = new[]
            {
                new[] {0d},
                new[] {1d},
                new[] {1d},
                new[] {0d},
                new[] {1d},
                new[] {0d},
                new[] {0d},
                new[] {1d}
            }
        };
    }
}