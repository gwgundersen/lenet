-- ============================================================================
-- Train LeNet5 classifier on MNIST.
-- ============================================================================

require 'svhn'
require 'lenet'
require 'Trainer'


-------------------------------------------------------------------------------
-- Parameters and constants.

local N_EPOCHS           = 2
local BATCH_SIZE         = 20
local CLASSES            = {'1','2','3','4','5','6','7','8','9','10'}

local GEOMETRY = {3, 32, 32}
local model = lenet.load(GEOMETRY)
local criterion = nn.ClassNLLCriterion()  -- Negative log likelihood.

-- Create training set and normalize
local trainData = svhn.loadTrainSet()
local testData = svhn.loadTestSet()

local trainer = Trainer(model, criterion, CLASSES, BATCH_SIZE, GEOMETRY)

for i = 1, N_EPOCHS do
    print('Epoch', i)
    trainer:train(trainData)
end

local confusion = trainer:test(testData)
print(confusion)
