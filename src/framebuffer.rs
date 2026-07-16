use arrayvec::ArrayVec;
use janus::{GpuResource, texture::TextureView};

pub const MAX_ATTACHMENTS: usize = 8;

pub fn bind_default() {
    bind(FramebufferId::default());
}

pub fn bind(framebuffer: impl Into<FramebufferId>) {
    janus::debug_assert_gl!();
    let framebuffer = framebuffer.into();
    unsafe {
        janus::gl::BindFramebuffer(janus::gl::FRAMEBUFFER, framebuffer.0);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FramebufferId(u32);
impl FramebufferId {
    pub fn bind(self) {
        crate::framebuffer::bind(self);
    }
}
impl From<FramebufferView> for FramebufferId {
    fn from(value: FramebufferView) -> Self {
        value.id
    }
}
impl From<&Framebuffer> for FramebufferId {
    fn from(value: &Framebuffer) -> Self {
        value.id
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FramebufferView {
    pub id: FramebufferId,
    pub width: u32,
    pub height: u32,
}
impl From<&Framebuffer> for FramebufferView {
    fn from(value: &Framebuffer) -> Self {
        Self::from_framebuffer(value)
    }
}
impl FramebufferView {
    pub const fn from_framebuffer(framebuffer: &Framebuffer) -> Self {
        Self {
            id: framebuffer.id,
            width: framebuffer.width,
            height: framebuffer.height,
        }
    }

    pub const fn id(&self) -> FramebufferId {
        self.id
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FramebufferError {
    #[error("default framebuffer was specified but it not yet defined")]
    Undefined,
    #[error("an attachment point of the framebuffer is incomplete")]
    IncompleteAttachment,
    #[error("framebuffer has no images attached to it")]
    IncompleteMissingAttachment,
    #[error("draw buffer color attachment is not a valid object type")]
    IncompleteDrawBuffer,
    #[error("read buffer color attachment is not a valid object type")]
    IncompleteReadBuffer,
    #[error("an attachment image produced an internal format error")]
    Unsupported,
    #[error("the number of samples is not equal for all attachments")]
    IncompleteMultisample,
    #[error(
        "framebuffer has a layered attachment but other attachments do not respect the necessary conditions"
    )]
    IncompleteLayerTargets,
}

pub type FramebufferResult = Result<Framebuffer, FramebufferError>;

#[derive(Debug)]
pub struct Framebuffer {
    pub id: FramebufferId,
    pub width: u32,
    pub height: u32,
    color_attachments: ArrayVec<TextureView, MAX_ATTACHMENTS>,
    depth_attachment: Option<TextureView>,
}
impl Framebuffer {
    pub fn new(
        width: u32,
        height: u32,
        color_attachments: &[TextureView],
        depth_attachment: Option<TextureView>,
    ) -> FramebufferResult {
        janus::debug_assert_gl!();

        let mut id = 0;
        unsafe {
            janus::gl::CreateFramebuffers(1, &mut id);
        };

        for (i, tex) in color_attachments.iter().enumerate() {
            unsafe {
                janus::gl::NamedFramebufferTexture(
                    id,
                    janus::gl::COLOR_ATTACHMENT0 + i as u32,
                    tex.resource_id(),
                    0,
                );
            }
        }
        if let Some(depth_texture) = &depth_attachment {
            unsafe {
                janus::gl::NamedFramebufferTexture(
                    id,
                    janus::gl::DEPTH_ATTACHMENT,
                    depth_texture.resource_id(),
                    0,
                );
            }
        }

        let status = unsafe { janus::gl::CheckNamedFramebufferStatus(id, janus::gl::FRAMEBUFFER) };
        if status != janus::gl::FRAMEBUFFER_COMPLETE {
            use FramebufferError::*;

            return match status {
                janus::gl::FRAMEBUFFER_UNDEFINED => Err(Undefined),
                janus::gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => Err(IncompleteAttachment),
                janus::gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => {
                    Err(IncompleteMissingAttachment)
                }
                janus::gl::FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER => Err(IncompleteDrawBuffer),
                janus::gl::FRAMEBUFFER_INCOMPLETE_READ_BUFFER => Err(IncompleteReadBuffer),
                janus::gl::FRAMEBUFFER_UNSUPPORTED => Err(Unsupported),
                janus::gl::FRAMEBUFFER_INCOMPLETE_MULTISAMPLE => Err(IncompleteMultisample),
                janus::gl::FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS => Err(IncompleteLayerTargets),
                0 => {
                    let mut last_error = 0;
                    #[allow(irrefutable_let_patterns, reason = "glGetError eventually returns 0")]
                    while let err = unsafe { janus::gl::GetError() } {
                        if err != 0 {
                            last_error = err;
                        } else {
                            break;
                        }
                    }
                    match last_error {
                        janus::gl::INVALID_ENUM => {
                            panic!("framebuffer creation unrecoverable error: invalid target")
                        }
                        janus::gl::INVALID_OPERATION => panic!(
                            "framebuffer creation error: invalid framebuffer name is not 0 or an existing framebuffer"
                        ),
                        _ => unreachable!(),
                    };
                }
                _ => unreachable!(),
            };
        }
        // status all ok !!

        let color_attachments = color_attachments.iter().copied().collect();
        Ok(Self {
            id: FramebufferId(id),
            width,
            height,
            color_attachments,
            depth_attachment,
        })
    }

    pub fn bind(&self) {
        crate::framebuffer::bind(self);
    }

    pub fn unbind() {
        crate::framebuffer::bind_default();
    }

    pub const fn id(&self) -> FramebufferId {
        self.id
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn as_view(&self) -> FramebufferView {
        FramebufferView::from_framebuffer(self)
    }
}
impl Drop for Framebuffer {
    fn drop(&mut self) {
        janus::debug_assert_gl!();
        unsafe {
            janus::gl::DeleteFramebuffers(1, &self.id.0);
        }
    }
}
