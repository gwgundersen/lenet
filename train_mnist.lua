-- ============================================================================
-- Train LeNet5 classifier on MNIST.
-- ============================================================================

require 'mnist'
require 'lenet'
require 'Trainer'


-------------------------------------------------------------------------------
-- Parameters and constants.

local N_EPOCHS           = 1
local BATCH_SIZE         = 20
local N_TRAINING_BATCHES = 60000
local N_TESTING_BATCHES  = 10000
local MEAN               = 0
local STD                = 1
local CLASSES            = {'1','2','3','4','5','6','7','8','9','10'}
local GEOMETRY           = {1, 32, 32 }
local model              = lenet.load(GEOMETRY)
local criterion          = nn.ClassNLLCriterion()  -- Negative log likelihood.

-- Create training set and normalize
local trainData = mnist.loadTrainSet(N_TRAINING_BATCHES, GEOMETRY)
trainData:normalizeGlobal(MEAN, STD)
local testData = mnist.loadTestSet(N_TESTING_BATCHES, GEOMETRY)
testData:normalizeGlobal(MEAN, STD)

local trainer = Trainer(model, criterion, CLASSES, BATCH_SIZE, GEOMETRY)

for i = 1, N_EPOCHS do
    print('Epoch', i)
    trainer:train(trainData)
end

local confusion = trainer:test(testData)
print(confusion)
