-- ============================================================================
-- LeNet 5.
-- ============================================================================

require 'cunn'
require 'cudnn'
require 'nn'


lenet = {}


-------------------------------------------------------------------------------
-- Parameters and constants.

local FILTER_SIZE          = 5
local POOL_SIZE            = 2
local STEP_SIZE            = 1
local PADDING              = 0  -- Use padding=2 if we want output to be same
                                -- size as input.
local N_FEATURE_MAPS_1ST_L = 6
local N_FEATURE_MAPS_2ND_L = 16
local WIDTH_FC_L           = 84
local N_CLASSES            = 10


-------------------------------------------------------------------------------
-- Construct LeNet convolutional neural network.

function lenet.new(geometry)

    local nChannels = geometry[1]  -- B&W=1, RGB=3.
    local imgWidth = geometry[2]
    local imgHeight = geometry[3]

    local model = nn.Sequential():cuda()

    -----------------------------------
    -- 2 ConvNet layers
    -- Arguments:
    --      input depth, output depth, kernel width, kernel height
    --      [step width], [step height], [padding width], [padding height]
    model:add(nn.SpatialConvolution(nChannels, N_FEATURE_MAPS_1ST_L,
        FILTER_SIZE, FILTER_SIZE, STEP_SIZE, STEP_SIZE, PADDING, PADDING))
    model:add(nn.SpatialMaxPooling(POOL_SIZE, POOL_SIZE, STEP_SIZE, STEP_SIZE))
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(N_FEATURE_MAPS_1ST_L, N_FEATURE_MAPS_2ND_L,
        FILTER_SIZE, FILTER_SIZE, STEP_SIZE, STEP_SIZE, PADDING, PADDING))
    model:add(nn.SpatialMaxPooling(POOL_SIZE, POOL_SIZE, STEP_SIZE, STEP_SIZE))
    model:add(nn.ReLU())

    -- Programmatically compute output size.
    local dummy_input  = torch.Tensor(nChannels, imgWidth, imgHeight)
    model:forward(dummy_input)
    local output_w = model.modules[6].output[1]:size()[1]  -- Convert tensor to value.
    local output_h = model.modules[6].output[2]:size()[1]
    print('<lenet> output (w, h):', output_w, output_h)

    -----------------------------------
    -- 1 fully-connected layer
    local newShape = N_FEATURE_MAPS_2ND_L * output_w * output_h
    model:add(nn.Reshape(newShape))
    model:add(nn.Linear(newShape, WIDTH_FC_L))
    model:add(nn.ReLU())

    -----------------------------------
    -- Output layer
    model:add(nn.Linear(WIDTH_FC_L, N_CLASSES))

    return model
end
