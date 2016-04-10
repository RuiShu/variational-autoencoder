local Logger = torch.class("Logger")
local c = require 'trepl.colorize'
require 'optim'
require 'utils.grid'
require 'image'

function Logger:__init(cmd)
   self.cmd = cmd
   if cmd.z_size == 2 then
      self.nRow = 40
      self.nCol = 40
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
   self.optLogger = optim.Logger('save/FooBar/FooBar.log')
   self.optLogger:setNames{'nelbo', 'kld', 'bce'}
   self.win = nil
   self.recordCount = 0
end

function Logger:receiveRecord(comm)
   -- comm is a {}
   self.recordCount = self.recordCount + 1
   self.kldStatus = self.kldStatus or comm.kldErr
   self.bceStatus = self.bceStatus or comm.bceErr
   self.nElboStatus = self.nElboStatus or comm.nElbo
   self.kldStatus = 0.99*self.kldStatus + 0.01*comm.kldErr
   self.bceStatus = 0.99*self.bceStatus + 0.01*comm.bceErr
   self.nElboStatus = 0.99*self.nElboStatus + 0.01*comm.nElbo
   self.optLogger:add{comm.nElbo/200, comm.kldErr/200, comm.bceErr/200}
   if self.recordCount % 50 == 0 and self.cmd.z_size == 2 then
      self:visualize(comm.decoder)
   end
end

function Logger:visualize(decoder)
   local recon = decoder:forward(self.code)
   local stack = grid.stack(recon:view(self.nRow*self.nCol,28,28),
                             self.nRow, self.nCol)
   self.win = image.display{image=stack, win=self.win}
end

function Logger:log()
   self.epoch = self.epoch or 0
   self.epoch = self.epoch + 1
   print(c.green 'Epoch: '..self.epoch)
   print(c.red '==> '..'Elbo: '..-self.nElboStatus/200)
   print(c.red '==> '..'KLD: '..self.kldStatus/200)
   print(c.red '==> '..'BCE: '..self.bceStatus/200)
   self.optLogger:style{'-','-','-'}
   self.optLogger:plot()
end

function Logger:cuda()
   if self.cmd.z_size == 2 then
      self.code = self.code:cuda()
   end
end
