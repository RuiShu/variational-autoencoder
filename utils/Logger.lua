local Logger = torch.class("Logger")
local c = require 'trepl.colorize'
require 'optim'
require 'utils.grid'
require 'image'

function Logger:__init(cmd)
   self.cmd = cmd
   if cmd.zSize == 2 and cmd.showVis then
      self.nRow = 10
      self.nCol = 14
      local dist = 2
      self.code = torch.Tensor(self.nRow*self.nCol,2)
      local x = torch.linspace(-dist,dist,self.nCol)
      local y = torch.linspace(-dist,dist,self.nRow)
      local idx = 0
      for yi = y:nElement(),1,-1 do
         for xi = 1,x:nElement() do
            idx = idx + 1
            self.code[{idx,1}] = x[xi]
            self.code[{idx,2}] = y[yi]
         end
      end
   end
   -- create logger
   self.optLogger = optim.Logger('save/'..cmd.model..'/'..cmd.model..'.log')
   self.optLogger.showPlot = false
   self.optLogger:setNames{'nelbo', 'kld', 'rec'}
   self.win = nil
   self.recordCount = 0
end

function Logger:receiveRecord(comm)
   -- comm is a {}
   self.recordCount = self.recordCount + 1
   self.kldStatus = self.kldStatus or comm.kldErr
   self.recStatus = self.recStatus or comm.recErr
   self.nElboStatus = self.nElboStatus or comm.nElbo
   self.kldStatus = 0.99*self.kldStatus + 0.01*comm.kldErr
   self.recStatus = 0.99*self.recStatus + 0.01*comm.recErr
   self.nElboStatus = 0.99*self.nElboStatus + 0.01*comm.nElbo
   self.optLogger:add{comm.nElbo/200, comm.kldErr/200, comm.recErr/200}
   if self.recordCount % 5 == 0 and self.cmd.zSize == 2 and self.cmd.showVis then
      self:visualize(comm.decoder)
   end
end

function Logger:visualize(decoder)
   local recon = decoder:forward(self.code)
   if recon:dim() == 3 then
      local recon = grid.split(recon, 2)
   end
   -- end of hack
   recon = recon:view(self.nRow*self.nCol, self.cmd.height, self.cmd.width)
   local stack = grid.stack(recon, self.nRow, self.nCol)
   self.win = image.display{image=stack, win=self.win, zoom=3}
end

function Logger:log()
   self.epoch = self.epoch or 0
   self.epoch = self.epoch + 1
   print(c.green 'Epoch: '..self.epoch)
   print(c.red '==> '..'KLD: '..self.kldStatus/200)
   print(c.red '==> '..'REC: '..self.recStatus/200)
   print(c.red '==> '..'nElbo: '..self.nElboStatus/200)
   -- self.optLogger:style{'-','-','-'}
   -- self.optLogger:plot()
end

function Logger:cuda()
   if self.cmd.zSize == 2 and self.cmd.showVis then
      self.code = self.code:cuda()
   end
end
